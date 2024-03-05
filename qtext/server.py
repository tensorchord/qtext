from __future__ import annotations

import falcon
import msgspec
from defspec import OpenAPI, RenderTemplate
from falcon import App, Request, Response

from qtext.engine import RetrievalEngine
from qtext.log import logger
from qtext.spec import (
    AddDocRequest,
    AddNamespaceRequest,
    DocResponse,
    HighlightRequest,
    HighlightResponse,
    QueryDocRequest,
)


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
        resp.text = "Check the OpenAPI spec at `/openapi/redoc`"


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


class HighlightResource:
    def __init__(self, engine: RetrievalEngine) -> None:
        self.engine = engine

    def on_post(self, req: Request, resp: Response):
        request = validate_request(HighlightRequest, req, resp)
        if request is None:
            return

        resp.data = msgspec.json.encode(self.engine.highlight(request))
        resp.content_type = falcon.MEDIA_JSON


class OpenAPIResource:
    def __init__(self):
        self.openapi = OpenAPI()
        self.openapi.register_route(
            "/api/namespace",
            "post",
            "Create a namespace",
            request_type=AddNamespaceRequest,
        )
        self.openapi.register_route(
            "/api/doc", "post", "Add a document", request_type=AddDocRequest
        )
        self.openapi.register_route(
            "/api/query",
            "post",
            "Get the similar documents",
            request_type=QueryDocRequest,
            response_type=list[DocResponse],
        )
        self.openapi.register_route(
            "/api/highlight",
            "post",
            "Highlight the semantic similar words in the documents according to the query",
            request_type=HighlightRequest,
            response_type=HighlightResponse,
        )
        self.spec = self.openapi.to_json()

    def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_JSON
        resp.data = self.spec


class OpenAPIRender:
    def __init__(self, spec_url: str, template: RenderTemplate) -> None:
        self.template = template.value.format(spec_url=spec_url)

    def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_HTML
        resp.text = self.template


def create_app(engine: RetrievalEngine) -> App:
    app = App()
    app.add_route("/", HealthCheck())
    app.add_route("/api/namespace", NamespaceResource(engine))
    app.add_route("/api/doc", DocResource(engine))
    app.add_route("/api/query", QueryResource(engine))
    app.add_route("/api/highlight", HighlightResource(engine))
    app.add_route("/openapi/spec.json", OpenAPIResource())
    app.add_route(
        "/openapi/swagger", OpenAPIRender("/openapi/spec.json", RenderTemplate.SWAGGER)
    )
    app.add_route(
        "/openapi/redoc", OpenAPIRender("/openapi/spec.json", RenderTemplate.REDOC)
    )
    app.add_route(
        "/openapi/scalar", OpenAPIRender("/openapi/spec.json", RenderTemplate.SCALAR)
    )
    app.add_error_handler(Exception, uncaught_exception_handler)
    return app
