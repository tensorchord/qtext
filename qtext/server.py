from __future__ import annotations

import falcon
import msgspec
from falcon import App, Request, Response

from qtext.log import logger
from qtext.spec import AddDocRequest, NamespaceRequest, QueryDocRequest


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
    def on_post(self, req: Request, resp: Response):
        request = validate_request(AddDocRequest, req, resp)
        if request is None:
            return


class QueryResource:
    def on_post(self, req: Request, resp: Response):
        request = validate_request(QueryDocRequest, req, resp)
        if request is None:
            return


class NamespaceResource:
    def on_post(self, req: Request, resp: Response):
        request = validate_request(NamespaceRequest, req, resp)
        if request is None:
            return


def create_app() -> App:
    app = App()
    app.add_route("/", HealthCheck())
    app.add_route("/api/namespace", NamespaceResource())
    app.add_route("/api/doc", DocResource())
    app.add_route("/api/query", QueryResource())
    app.add_error_handler(Exception, uncaught_exception_handler)
    return app
