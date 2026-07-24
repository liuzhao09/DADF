
import os
import sys
import atexit
from datetime import datetime

class _TeeLogger:

    def __init__(self, log_path: str, terminal):
        self._terminal = terminal
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._file = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, message):
        self._terminal.write(message)

        self._file.write(message.replace('\r', '\n'))
        self._file.flush()

        try:
            self._terminal.flush()
        except Exception:
            pass

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        if sys.stdout is self:
            sys.stdout = self._terminal
        try:
            self._file.close()
        except Exception:
            pass

    @property
    def encoding(self):
        return self._terminal.encoding

    @property
    def errors(self):
        return self._terminal.errors

    def isatty(self):
        return self._terminal.isatty()

    def fileno(self):
        return self._terminal.fileno()

def setup_logger(log_dir: str, args=None, script: str = '') -> str:

    prev = sys.stdout
    if isinstance(prev, _TeeLogger):
        prev.close()

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_path = os.path.join(log_dir, '{}_pid{}.log'.format(ts, os.getpid()))

    terminal = sys.stdout if not isinstance(sys.stdout, _TeeLogger) else sys.__stdout__
    tee      = _TeeLogger(log_path, terminal)
    sys.stdout = tee

    atexit.register(tee.close)

    line = '=' * 72
    print(line)
    print('Script  : {}'.format(script or 'unknown'))
    print('Time    : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('Log     : {}'.format(log_path))
    print(line)
    if args is not None:
        print('Arguments:')
        for k, v in sorted(vars(args).items()):
            rv = repr(v)
            if len(rv) > 120:
                rv = rv[:117] + '...'
            print('  {:20s} = {}'.format(k, rv))
    print(line)
    print()

    return log_path
