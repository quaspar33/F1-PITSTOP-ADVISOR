from kedro.pipeline import Node, Pipeline

from .nodes import split_data, fit_model_search


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_data,
                inputs=["data", "params:regressors.target_label"],
                outputs=["X", "y"],
                name="split_data_node",
            ),
            Node(
                func=fit_model_search,
                inputs=["X", "y"],
                outputs="regressor",
                name="fit_model_search_node",
            ),
        ]
    )
