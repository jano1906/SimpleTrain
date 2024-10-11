import functools
import logging
import os
import sys
from typing import Optional
from collections import defaultdict, deque
import datetime
import time
import torch

logger = logging.getLogger("SimpleTrain")

class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, num: int, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, num)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                eta_seconds = iter_time.avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {} ({:.6f} s / it)".format(header, total_time_str, total_time / n_iterations))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
):
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
) -> None:
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        capture_warnings: Whether warnings should be captured as logs.
    """
    logging.captureWarnings(capture_warnings)
    _configure_logger(name, level=level, output=output)
