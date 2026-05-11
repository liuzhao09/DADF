"""
实验日志系统

用法（在每个 run_*.py 的 main 入口处调用）:
    from logger import setup_logger
    setup_logger('logs/baseline/egmn', args, script='run_egmn.py')

功能：
  - 将 stdout 同步输出到控制台 + 日志文件（tee 模式）
  - 日志文件按时间戳命名：logs/{category}/{model}/YYYYMMDD_HHMMSS.log
  - 日志顶部自动打印完整训练参数
  - atexit 自动关闭文件，防止资源泄露
  - 多次调用时正确释放旧的文件句柄
"""

import os
import sys
import atexit
from datetime import datetime


class _TeeLogger:
    """将 stdout 同时输出到终端和文件（tee 模式）"""

    def __init__(self, log_path: str, terminal):
        self._terminal = terminal
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._file = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, message):
        self._terminal.write(message)
        # 将 \r（进度覆写）替换为 \n 写入文件，避免日志乱码
        self._file.write(message.replace('\r', '\n'))
        self._file.flush()
        # 同时 flush 原 stdout，保证 shell 重定向（如 `python ... > run.log`）也能实时看到
        try:
            self._terminal.flush()
        except Exception:
            pass

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        """关闭文件并还原 sys.stdout"""
        if sys.stdout is self:
            sys.stdout = self._terminal
        try:
            self._file.close()
        except Exception:
            pass

    # ── io.TextIOBase 协议补全（兼容 torch.distributed、tqdm 等）──
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
    """
    初始化日志：重定向 stdout 到 tee logger，打印参数头。

    Args:
        log_dir : 日志目录，如 'logs/baseline/egmn'
        args    : argparse.Namespace，用于打印训练参数
        script  : 脚本名，如 'run_egmn.py'

    Returns:
        str: 日志文件完整路径
    """
    # 若已存在旧的 _TeeLogger，先还原并关闭（防止多次调用时文件句柄泄露）
    prev = sys.stdout
    if isinstance(prev, _TeeLogger):
        prev.close()

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 加 PID 避免并行实验同秒启动时文件名冲突（否则后启动的会 truncate 先启动的内容）
    log_path = os.path.join(log_dir, '{}_pid{}.log'.format(ts, os.getpid()))

    # 取当前真实终端（可能已被还原）
    terminal = sys.stdout if not isinstance(sys.stdout, _TeeLogger) else sys.__stdout__
    tee      = _TeeLogger(log_path, terminal)
    sys.stdout = tee

    # 进程退出时自动关闭（兼容异常/OOM 退出场景）
    atexit.register(tee.close)

    # ── 日志头 ───────────────────────────────────────────────
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
