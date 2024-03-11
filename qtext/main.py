import waitress

from qtext.config import Config
from qtext.engine import RetrievalEngine
from qtext.log import logger
from qtext.server import create_app


def run():
    config = Config.with_config_file()
    logger.setLevel(config.server.log_level)
    logger.info(config)
    logger.info("starting the server")
    engine = RetrievalEngine(config=config)
    app = create_app(engine)
    waitress.serve(app, host="0.0.0.0", port=8000)
