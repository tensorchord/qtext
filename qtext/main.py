import waitress

from qtext.config import Config
from qtext.engine import RetrievalEngine
from qtext.log import logger
from qtext.server import create_app


def run():
    logger.info("starting the server")
    engine = RetrievalEngine(Config())
    app = create_app(engine)
    waitress.serve(app, host="0.0.0.0", port=8000)
