if exists('g:autoloaded_lama')
  finish
endif
let g:autoloaded_lama = 1

scriptencoding utf-8

let s:root = expand('<sfile>:h:h')
let s:home = expand(expand('$HOME'))
let s:home = substitute(s:home, '\', '/', 'g')
let s:config_file = s:home . '/.config/lama/config.vim'

if filereadable(s:config_file)
  source $HOME/.config/lama/config.vim
else
  " Create file
  call mkdir(s:home . '/.config/lama', 'p')
  call writefile(['" This is the configuration file for lama.'], s:config_file)
  call writefile(["\"let g:lama_ws_url = 'ws://localhost:7860/queue/join'"], s:config_file, "a")
endif

function! lama#Stop()
  if exists('b:lama')
    call s:flush()
    echo "Stopping Lama"
    call jobstop(b:lama.job)
  endif
endfunction

function! lama#Toggle()
  if exists('b:lama')
    call s:flush()
    echo "Stopping Lama"
    call jobstop(b:lama.job)
  else
    let lama_url = get(g:, "lama_ws_url", "")
    if empty(lama_url)
      echo "Lama URL not set. Please set g:lama_ws_url in " . s:config_file
      return
    endif
    echo "Starting Lama"
    let b:lama = {}
    let job = jobstart(['node', s:root . '/lama/helper.js', g:lama_ws_url], {
          \'on_stdout': function('s:onOut'),
          \'on_exit':function('s:onExit'),
        \})

    let b:lama.job = job

    let prompt = join(getline(1, '$'), "\n")
    let b:lama.prompt = prompt

    call chansend(job, prompt)
  endif
endfunction

let s:hlgroup = 'LamaSuggestion'

function! lama#NvimNs() abort
  return nvim_create_namespace('varal7-lama')
endfunction

function! s:onOut(job, text, event)
  " display the suggestion as virtual text
  
  let text = a:text
  " remove the trailing newline
  if empty(text[-1])
    call remove(text, -1)
  endif
  if empty(text)
    return
  endif


  let text = join(text, "\n")
  "if text doesn't start with <START>, then it's a message
  if text[0:7] != '<START> '
    echo text
    return
  endif
  "otherwise it's a suggestion
  "split at <START> because the channel might have sent multiple messages in
  let text = split(text, '<START> ')[-1]

  let b:lama.suggestion = text

  let prompt = b:lama.prompt
  let remaining_text = strpart(text, len(prompt))
  let text = split(remaining_text, "\n")
  if empty(text)
    return
  endif

  " display virtual text
  let data = {'id': 1}
  let lastline = line('$')    " Get line number of last line in buffer
  let linewidth = winwidth('%')
  let eol_col = strlen(getline(lastline)) + 1  " Get column number of end-of-line

  " First we break the first line if it goes over the width of the window
  if eol_col + strlen(text[0]) > linewidth
    let split = linewidth - eol_col - 3
    let data.virt_text = [[strpart(text[0], 0, split), s:hlgroup]]
    let text[0] = strpart(text[0], split)
  else
    let data.virt_text = [[text[0], s:hlgroup]]
    call remove(text, 0)
  endif

  " Then we add the rest of the lines, by breaking them if they go over the width of the window
  let data.virt_lines = []
  for line in text
    while strlen(line) > linewidth
      call add(data.virt_lines, [[strpart(line, 0, linewidth), s:hlgroup]])
      let line = strpart(line, linewidth)
    endwhile
    call add(data.virt_lines, [[line, s:hlgroup]])
  endfor

  let data.hl_mode = 'combine'
  let data.virt_text_pos = 'overlay'
  call nvim_buf_set_extmark(0, lama#NvimNs(), lastline-1, eol_col-1, data)
endfunction 


function! lama#ShowVirtualText()
  let lastline = line('$')    " Get line number of last line in buffer
  let eol_col = strlen(getline(lastline)) + 1  " Get column number of end-of-line
  let virt_text = 'End of buffer'             " Virtual text to display
  let data = {'id': 1}
  let data.virt_text = [["Hello", s:hlgroup]]
  let data.hl_mode = 'combine'
  let data.virt_text_pos = 'overlay'
  echom lastline
  echom eol_col

  call nvim_buf_set_extmark(0, lama#NvimNs(), lastline-1, eol_col-1, data)
endfunction

function! s:onExit(job, data, type)
  call s:flush()
  unlet! b:lama
endfunction


function! s:flush()
  if exists('b:lama')
    call nvim_buf_clear_namespace(0, lama#NvimNs(), 0, -1)
    let text = get(b:lama, 'suggestion', "")
    if empty(text)
      return
    endif
    let text = split(text, "\n")
    call append(0, text)
    call deletebufline("%", len(text) + 1, "$")
  endif
endfunction