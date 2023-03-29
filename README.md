# Lama.vim

```
English: a lion
French: un lion

English: a giraffe
French: une girafe

English: a llama
French: un lama
```

## Installation

Use your favorite plugin manager. E.g. with Plug:

```
Plug 'varal7/lama.vim', {'do': { -> lama#install()}}
```

If not using a plugin manager, please run `yarn` or `npm install` in the `lama` folder.

Configure `g:lama_ws_url` to point to a Gradio websocket (should start with `ws://` or `wss://` and end in `/join/queue/`)

For example, by using [https://github.com/oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

## Usage

By default, keybindings are Ctrl-L for start/stop in insert/normal mode and only in text files.
To map to Ctrl-X, use e.g.

```
imap <C-x> <Plug>(lama-toggle)
```
