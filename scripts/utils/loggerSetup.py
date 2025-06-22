import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from colorama import init, Fore, Style
from utils.lunaConfig import LunaConfig

# Initialize colorama for colored console logs
init(autoreset=True)

# Load environment variables (e.g., MAXBYTES_LOGGER, BACKUPCOUNT_LOGGER)
load_dotenv()


class ColoredFormatter(logging.Formatter):
    """ Custom formatter to add colors to log levels """
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Fore.RESET}"
        return super().format(record)


class InitLoggers:
    """
    A logger initializer that supports:
    - Rotating file logging
    - Optional console or error-only stream logging
    - Color-coded log levels for console output
    - Prevents duplicate handlers
    """

    def __init__(self, name, max_bytes, backup_count, logfile_name="main.log", save_path=f"{LunaConfig.DATA_DIR}/logs/", console_output=True):
        self.formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)8s:%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent propagation to root logger

        os.makedirs(save_path, exist_ok=True)
        log_file_path = os.path.abspath(os.path.join(save_path, logfile_name))

        # ✅ Prevent duplicate file handlers
        if not any(
            isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == log_file_path
            for h in self.logger.handlers
        ):
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=int(max_bytes),
                backupCount=int(backup_count),
                encoding='utf-8'
            )
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        # ✅ Add console handler only once
        if console_output:
            if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(self.formatter)
                console_handler.setLevel(logging.DEBUG)
                self.logger.addHandler(console_handler)
        else:
            if not any(isinstance(h, logging.StreamHandler) and h.level == logging.ERROR for h in self.logger.handlers):
                error_handler = logging.StreamHandler()
                error_handler.setFormatter(self.formatter)
                error_handler.setLevel(logging.ERROR)
                self.logger.addHandler(error_handler)


def get_logger(name, logfile_name, console_output=True):
    """
    Retrieve a properly configured logger instance.

    :param name: Logger name
    :param logfile_name: Log file name
    :param console_output: Whether to log to console (True) or only errors (False)
    :return: Logger instance
    """
    max_bytes = os.getenv("MAXBYTES_LOGGER", 1048576)  # 1MB default
    backup_count = os.getenv("BACKUPCOUNT_LOGGER", 5)
    logger_instance = InitLoggers(
        name=name,
        max_bytes=max_bytes,
        backup_count=backup_count,
        logfile_name=logfile_name,
        save_path=os.path.join(LunaConfig.DATA_DIR, "logs"),
        console_output=console_output
    ).logger
    return logger_instance
