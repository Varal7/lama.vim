if exists('g:autoloaded_lama')
  finish
endif
let g:autoloaded_lama = 1

scriptencoding utf-8

let s:root = expand('<sfile>:h:h')
let s:js_root = s:root . '/lama'
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

function! lama#Stop() abort
  if exists('b:lama')
    call s:flush()
    echo "Stopping Lama"
    call jobstop(b:lama.job)
  endif
endfunction

function! lama#Toggle() abort
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
          \'on_stderr': function('s:onErr'),
          \'on_exit':function('s:onExit'),
        \})

    let b:lama.job = job

    " Get position of cursor
    let b:lama.line = line('.')
    let b:lama.col = col('.')

    " Get current buffer up to cursor
    let b:lama.prompt = join(getline(1, line('.') - 1), "\n")
    let b:lama.prompt .= "\n" . getline(line('.'))[0:col('.') - 2]

    call chansend(job, b:lama.prompt)
  endif
endfunction

let s:hlgroup = 'LamaSuggestion'

function! lama#NvimNs() abort
  return nvim_create_namespace('varal7-lama')
endfunction

function! lama#Install() abort
  let yarncmd = get(g:, 'lama_install_yarn_cmd', executable('yarnpkg') ? 'yarnpkg' : executable('yarn') ? 'yarn' : 'npm')
  let cmd = yarncmd . ' install'
  let cwd = s:js_root
  let job = jobstart(cmd, {
        \'on_stdout': function('s:onOut'),
        \'on_stderr': function('s:onErr'),
        \'on_exit':function('s:onExit'),
        \'cwd': cwd,
      \})
endfunction

fun s:error(msg) abort
   echohl Error
   for line in a:msg->split('\n')
       echom line
   endfor
   echohl None
endfun

function! s:onErr(job, text, event) abort
  if empty(a:text[-1])
    call remove(a:text, -1)
  endif
  if empty(a:text)
    return
  endif
  echon join(a:text, "\n")
  echoerr "Did you run call lama#Install()?"
endfunction

function! s:onOut(job, text, event) abort
  " display the suggestion as virtual text
  if !exists('b:lama')
    return
  endif
  
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
    echo "Lama: " . text
    return
  endif

  "otherwise it's a suggestion
  if !get(b:lama, "generating", 0)
    let b:lama.generating = 1
    echo "Generating suggestion... toggle again to stop"
  endif
  
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
  let curline = b:lama.line
  let curcol = b:lama.col
  let linewidth = winwidth('%') - 5

  " First we break the first line if it goes over the width of the window
  if curcol + strlen(text[0]) > linewidth
    let split = linewidth - curcol - 3
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
  call nvim_buf_set_extmark(0, lama#NvimNs(), curline-1, curcol-1, data)
endfunction 


function! lama#ShowVirtualText()
  let b:line = line('.')    " Get line number of last line in buffer
  let b:col = col('.')
  let data = {'id': 1}
  let data.virt_text = [["Hello", s:hlgroup]]
  let data.hl_mode = 'combine'
  let data.virt_text_pos = 'overlay'

  call nvim_buf_set_extmark(0, lama#NvimNs(), b:line-1, b:col-1, data)
endfunction

function! s:onExit(job, data, type)
  call s:flush()
  echo "Lama stopped"
  unlet! b:lama
endfunction


function! s:flush()
  if exists('b:lama') && !get(b:lama, 'flushed', 0)
    let b:lama.flushed = 1
    let text = get(b:lama, 'suggestion', "")
    let prompt = b:lama.prompt
    let line = b:lama.line

    call nvim_buf_clear_namespace(0, lama#NvimNs(), 0, -1)

    if empty(text)
      return
    endif

    let remaining_text = strpart(text, len(prompt))
    let text = split(remaining_text, "\n")

    let last_prompt_line = split(prompt, "\n")[-1]
    let first_line = last_prompt_line . text[0]

    " insert the suggestion
    call setline(line, first_line)

    " insert the rest of the suggestion
    call append(line, text[1:])

    let b:debug = text

    " move the cursor to the end of the suggestion
    let last_line = line + len(text) - 1
    let last_line_len = strlen(text[-1])
    call cursor(last_line, last_line_len + 1)

    " call append(0, text)
  endif
endfunction
