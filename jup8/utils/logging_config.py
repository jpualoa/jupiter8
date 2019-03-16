import logging, logging.handlers, traceback, os, yaml

# The following line suppreses the 'No handlers could be found for logger' error
# if no one calls configure_logging()
logging.getLogger('').addHandler(logging.NullHandler())

def configure(level=None, fmt=None, logfn=None, configfn=None):
    """Configures logging infrastructure with a StreamHandler that will output to
    the console and an optional RotatingFileHandler that will output to 'logfn'.
    Should only be called once within an application, i.e., inside __main__.
    
    Args:
      level:    log level (default: INFO)
      logfmt:   logging format string (default:
                '%(asctime)s [%(levelname)-8s] (%(process)d) %(module)12s :%(lineno)4s | %(message)s')
      logfn:    log file (will append if file already exists)
      configfn: Config file used to customize the logger (yaml)

    Config file format:

      logging:
          # log level
          log_level: <DEBUG|INFO|WARN|ERROR|CRITICAL>
          # log format
          log_format: <logging format string>
          # log to file (in addition to console)
          log_file: <log file>
    """
    LOGGING_LEVELS = {'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARN': logging.WARN,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}

    # Intialize defaults
    log_level = logging.INFO
    log_format = '%(asctime)s [%(levelname)-8s] (%(process)d) %(module)12s :%(lineno)4s | %(message)s'
    log_file = None

    # Parse configuration file
    if configfn is not None:
        try:
            with open(configfn) as f:
                config = yaml.load(stream=f)
                if 'logging' in config:
                    if 'log_level' in config:
                        key = config['logging']['log_level']
                        try:
                            log_level = LOGGING_LEVELS[str.upper(key)]
                        except:
                            print '[WARNING]: Could not interpret requested log level:', key
                            print 'Legal values are:', LOGGING_LEVELS.keys()
                    if 'log_format' in config['logging']:
                        log_format = config['logging']['log_format']
                    if 'log_file' in config['logging']:
                        log_file = config['logging']['log_file']
        except:
            print '[WARNING]: Problem parsing configuration file: ', configfn
            print traceback.format_exc()

    # Override settings with values explicitly passed in as arguments
    if level is not None: log_level = level
    if fmt   is not None: log_format = fmt
    if logfn is not None: log_file = logfn
   
    # Set up output directory for log files, if needed
    if log_file is not None:
        try:
            log_dir = os.path.dirname(log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        except:
            print 'Problem setting up log directory'
            print traceback.format_exc()
            raise

    # Configure basic logger
    formatter = logging.Formatter(log_format)
    logging.getLogger('').setLevel(log_level)

    # Configure file logger, if needed
    if log_file is not None: # we want to log to a file
        # no specific file name was provided, add a rotating logger
        fh = logging.handlers.RotatingFileHandler(log_file, mode='a',
          backupCount=15, maxBytes=1e8)
        fh.setFormatter(formatter)
        logging.getLogger('').addHandler(fh)

    # Configure console logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger('').addHandler(ch)
