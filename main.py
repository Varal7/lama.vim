import gradio as gr
import transformers
import torch
import time

import argparse

import gc
from queue import Queue
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM

# Read args

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='distilgpt2')
parser.add_argument('--fp16', action='store_true')

args = parser.parse_args()

# Load model

print('Loading tokenizer from {}'.format(args.model))
tokenizer = AutoTokenizer.from_pretrained(args.model)
print('Loading model from {}'.format(args.model))
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if args.fp16 else torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streaming utils

# Copied from https://github.com/PygmalionAI/gradio-ui/
class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc=func
        self.c_callback=callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True,None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()


# Gradio Interface
textbox = gr.Textbox(lines=25, label='Input')

output = gr.Textbox(lines=25, label='Output')

gradio = {}
default_params = {
    'max_new_tokens': 200,
    'do_sample': False,
    'temperature': 0.5,
    'top_p': 0.9,
    'typical_p': 1,
    'repetition_penalty': 1.05,
    'encoder_repetition_penalty': 1.0,
    'top_k': 0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 1,
    'early_stopping': False,
}

with gr.Row():
    with gr.Column():
        with gr.Box():
            gr.Markdown('Custom generation parameters ([reference](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig))')
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(0.01, 1.99, value=default_params['temperature'], step=0.01, label='temperature')
                    top_p = gr.Slider(0.0,1.0,value=default_params['top_p'],step=0.01,label='top_p')
                    top_k = gr.Slider(0,200,value=default_params['top_k'],step=1,label='top_k')
                    typical_p = gr.Slider(0.0,1.0,value=default_params['typical_p'],step=0.01,label='typical_p')
                with gr.Column():
                    repetition_penalty = gr.Slider(1.0, 1.5, value=default_params['repetition_penalty'],step=0.01,label='repetition_penalty')
                    encoder_repetition_penalty = gr.Slider(0.8, 1.5, value=default_params['encoder_repetition_penalty'],step=0.01,label='encoder_repetition_penalty')
                    no_repeat_ngram_size = gr.Slider(0, 20, step=1, value=default_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                    min_length = gr.Slider(0, 2000, step=1, value=0, label='min_length', interactive=False)
            do_sample = gr.Checkbox(value=default_params['do_sample'], label='do_sample')
    with gr.Column():
        with gr.Box():
            gr.Markdown('Contrastive search')
            penalty_alpha = gr.Slider(0, 5, value=default_params['penalty_alpha'], label='penalty_alpha')

        with gr.Box():
            gr.Markdown('Beam search (uses a lot of VRAM)')
            with gr.Row():
                with gr.Column():
                    num_beams = gr.Slider(1, 20, step=1, value=default_params['num_beams'], label='num_beams')
                with gr.Column():
                    length_penalty = gr.Slider(-5, 5, value=default_params['length_penalty'], label='length_penalty')
            early_stopping = gr.Checkbox(value=default_params['early_stopping'], label='early_stopping')

max_new_tokens = gr.Slider(minimum=1, maximum=2048, step=1, label='max_new_tokens', value=200)



def generate_reply(question,
                    max_new_tokens,
                    do_sample,
                    temperature,
                    top_p,
                    typical_p,
                    repetition_penalty,
                    encoder_repetition_penalty,
                    top_k,
                    min_length,
                    no_repeat_ngram_size,
                    num_beams,
                    penalty_alpha,
                    length_penalty,
                    early_stopping,
                    eos_token=None,
                    stopping_string=None):


    t0 = time.time()

    input_ids = tokenizer.encode(str(question), return_tensors='pt').to(device)

    output = input_ids[0]
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    if eos_token is not None:
        eos_token_ids.append(int(tokenizer.encode(eos_token)[0][-1]))


    stopping_criteria_list = transformers.StoppingCriteriaList()

    if stopping_string is not None:
        # Copied from https://github.com/PygmalionAI/gradio-ui/blob/master/src/model.py
        t = tokenizer.encode(stopping_string, 0, add_special_tokens=False, return_tensors="pt")
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=t, starting_idx=len(input_ids[0])))

    generate_params = {}

    generate_params.update({
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_ids,
        "stopping_criteria": stopping_criteria_list,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "typical_p": typical_p,
        "repetition_penalty": repetition_penalty,
        "encoder_repetition_penalty": encoder_repetition_penalty,
        "top_k": top_k,
        "min_length": 0,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beams": num_beams,
        "penalty_alpha": penalty_alpha,
        "length_penalty": length_penalty,
        "early_stopping": early_stopping,
    })

    generate_params.update({'inputs': input_ids})


    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            reply = tokenizer.decode(output)
            # Waiting for https://github.com/huggingface/transformers/pull/22402 to fix the issue with the tokenizer
            reply = reply.split(question)[-1]
            reply = question + reply


            if output[-1] in eos_token_ids:
                break
            yield reply

        yield reply

    t1 = time.time()
    print(f'Generation took {t1-t0:.2f} seconds')


interface = gr.Interface(
    fn=generate_reply,
    inputs =  [
        textbox,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        typical_p,
        repetition_penalty,
        encoder_repetition_penalty,
        top_k,
        min_length,
        no_repeat_ngram_size,
        num_beams,
        penalty_alpha,
        length_penalty,
        early_stopping,
        #  eos_token,
        #  stopping_string
    ],
    outputs=[output],
    allow_flagging="never"
)

interface.queue().launch()
