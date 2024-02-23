import waitress

from qtext.log import logger
from qtext.server import create_app


def run():
    logger.info("starting the server")
    app = create_app()
    waitress.serve(app, host="0.0.0.0", port=8000)
