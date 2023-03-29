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

Use your favorite plugin manager.
But also, please do `yarn` or `npm install` in the `lama` folder.

Configure `g:lama_ws_url` to point to a Gradio websocket (should start with `ws://` or `wss://` and end in `/join/queue/`)

## Usage

By default, keybindings are Ctrl-L for start/stop in insert/normal mode and only in text files.
To map to Ctrl-X, use e.g.

```
imap <C-x> <Plug>(lama-toggle)
```
