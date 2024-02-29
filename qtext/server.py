from __future__ import annotations

import falcon
import msgspec
from falcon import App, Request, Response

from qtext.engine import RetrievalEngine
from qtext.log import logger
from qtext.spec import AddDocRequest, AddNamespaceRequest, QueryDocRequest


def validate_request(spec: type[msgspec.Struct], req: Request, resp: Response):
    buf = req.stream.read()
    try:
        request = msgspec.json.decode(buf, type=spec)
    except (msgspec.ValidationError, msgspec.DecodeError) as err:
        logger.info(
            "failed to decode the request '%s' body %s: %s", req.path, spec, err
        )
        resp.status = falcon.HTTP_422
        resp.text = f"Validation error: {err}"
        resp.content_type = falcon.MEDIA_TEXT
        return None
    return request


def uncaught_exception_handler(
    req: Request, resp: Response, exc: Exception, params: dict
):
    logger.warning(
        "exception from endpoint '%s'",
        req.path,
        exc_info=exc,
    )
    raise falcon.HTTPError(falcon.HTTP_500)


class HealthCheck:
    def on_get(self, req: Request, resp: Response):
        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_TEXT


class DocResource:
    def __init__(self, engine: RetrievalEngine) -> None:
        self.engine = engine

    def on_post(self, req: Request, resp: Response):
        request = validate_request(AddDocRequest, req, resp)
        if request is None:
            return

        self.engine.add_doc(request)


class QueryResource:
    def __init__(self, engine: RetrievalEngine) -> None:
        self.engine = engine

    def on_post(self, req: Request, resp: Response):
        request = validate_request(QueryDocRequest, req, resp)
        if request is None:
            return

        docs = self.engine.query(request)
        resp.data = msgspec.json.encode(docs)
        resp.content_type = falcon.MEDIA_JSON


class NamespaceResource:
    def __init__(self, engine: RetrievalEngine) -> None:
        self.engine = engine

    def on_post(self, req: Request, resp: Response):
        request = validate_request(AddNamespaceRequest, req, resp)
        if request is None:
            return
        self.engine.add_namespace(request)


def create_app(engine: RetrievalEngine) -> App:
    app = App()
    app.add_route("/", HealthCheck())
    app.add_route("/api/namespace", NamespaceResource(engine))
    app.add_route("/api/doc", DocResource(engine))
    app.add_route("/api/query", QueryResource(engine))
    app.add_error_handler(Exception, uncaught_exception_handler)
    return app
