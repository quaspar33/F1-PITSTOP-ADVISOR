from kedro.pipeline import Node, Pipeline

from .nodes import load_sessions, prepare_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_sessions,
                inputs=None,
                outputs="sessions",
                name="session_node"
            ),
            Node(
                func=prepare_data,
                inputs="sessions",
                outputs="data",
                name="data_node",
            ),
        ]
    )
