import WebSocket from 'ws';

const randomHash = () => Math.random().toString(36).substr(2, 9);

const session = randomHash();

const params = {
  max_new_tokens: 200,
  do_sample: true,
  temperature: 1.99,
  top_p: 0.18,
  typical_p: 1,
  repetition_penalty: 1.15,
  top_k: 30,
  min_length: 0,
  no_repeat_ngram_size: 0,
  num_beams: 1,
  penalty_alpha: 0,
  length_penalty: 1,
  early_stopping: false,
};


async function main(context, args) {
  const ws = new WebSocket(args[0]);

  ws.on('message', async (message) => {
    const content = JSON.parse(message);
    if (content.msg === 'send_hash') {
      ws.send(
        JSON.stringify({
          session_hash: session,
          fn_index: 7,
        })
      );
    } else if (content.msg === 'estimation') {
      console.log(`${content.rank}/${content.queue_size}`)
    } else if (content.msg === 'send_data') {
      ws.send(
        JSON.stringify({
          session_hash: session,
          fn_index: 7,
          data: [
            context,
            params.max_new_tokens,
            params.do_sample,
            params.temperature,
            params.top_p,
            params.typical_p,
            params.repetition_penalty,
            params.top_k,
            params.min_length,
            params.no_repeat_ngram_size,
            params.num_beams,
            params.penalty_alpha,
            params.length_penalty,
            params.early_stopping,
          ],
        })
      );
    } else if (content.msg === 'process_generating') {
      console.log("<START>", content.output.data[0]);
    } else if (content.msg === 'process_completed') {
      console.log("<START>", content.output.data[0]);
      ws.close();
      process.exit()
    }
  });
}

const args = process.argv.slice(2);
if (args.length < 1) {
  console.log("Usage: node helper.js [wsUrl]");
  process.exit(1);
}

process.stdin.resume();

process.stdin.on('data', (data) => {
    if (data instanceof Buffer) {
      data = data.toString();
  }
  main(data, args);
});
