import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class LinearRegressionModel:
    def __init__(self) -> None:
        self.fit_intercept: Optional[bool] = None
        self.rcond: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.weights_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False
        self.training_details_: Dict[str, Any] = {}

    def build_model(
        self,
        fit_intercept: bool,
        rcond: float
    ) -> "LinearRegressionModel":
        if rcond <= 0:
            raise ValueError("rcond phải là số thực dương.")

        self.fit_intercept = fit_intercept
        self.rcond = rcond
        self.coef_ = None
        self.intercept_ = 0.0
        self.weights_ = None
        self.is_fitted = False
        self.training_details_ = {}
        return self

    def _to_numpy(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X_arr = np.array(X, dtype=np.float64)
            if not self.feature_names_:
                self.feature_names_ = [f"x_{i}" for i in range(X_arr.shape[1])]
        else:
            raise TypeError("X phải là numpy.ndarray hoặc pandas.DataFrame.")

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float64).ravel()
            elif isinstance(y, np.ndarray):
                y_arr = np.array(y, dtype=np.float64).ravel()
            else:
                raise TypeError("y phải là numpy.ndarray, pandas.Series hoặc DataFrame.")

            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError("Số dòng của X và y không khớp nhau.")

        return X_arr, y_arr

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> "LinearRegressionModel":
        if self.fit_intercept is None or self.rcond is None:
            raise RuntimeError("Vui lòng gọi build_model(fit_intercept, rcond) trước khi fit!")

        X_mat, y_vec = self._to_numpy(X, y)
        n_samples, n_features = X_mat.shape
        self.n_features_in_ = n_features

        if n_samples < n_features:
            raise ValueError(f"Số mẫu ({n_samples}) nhỏ hơn số đặc trưng ({n_features}).")

        if self.fit_intercept:
            bias_col = np.ones((n_samples, 1), dtype=np.float64)
            X_design = np.hstack([bias_col, X_mat])
        else:
            X_design = X_mat

        w_star, _, rank, s = np.linalg.lstsq(X_design, y_vec, rcond=self.rcond)

        if self.fit_intercept:
            self.intercept_ = float(w_star[0])
            self.coef_ = w_star[1:].astype(np.float64)
            self.weights_ = w_star.astype(np.float64)
        else:
            self.intercept_ = 0.0
            self.coef_ = w_star.astype(np.float64)
            self.weights_ = w_star.astype(np.float64)

        y_train_pred = (np.dot(X_mat, self.coef_) + self.intercept_).ravel()
        res = y_vec - y_train_pred
        rss = float(np.sum(res ** 2))
        tss = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
        cond_number = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")

        self.training_details_ = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "design_matrix_shape": list(X_design.shape),
            "effective_rank": int(rank),
            "condition_number": round(cond_number, 4),
            "singular_values": [round(float(v), 4) for v in s],
            "RSS": round(rss, 4),
            "TSS": round(tss, 4)
        }

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")

        X_mat, _ = self._to_numpy(X)
        if X_mat.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Kích thước đặc trưng không khớp: Cần {self.n_features_in_}, nhận được {X_mat.shape[1]}."
            )

        return (np.dot(X_mat, self.coef_) + self.intercept_).ravel()

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Dict[str, float]:
        _, y_true = self._to_numpy(X, y)
        y_pred = self.predict(X)

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))) * 100.0

        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE(%)": round(mape, 4),
            "R2": round(r2, 4)
        }

    def get_training_details(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit!")
        return self.training_details_

    def get_model_parameters(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit!")
        weights_dict = {
            self.feature_names_[i]: round(float(self.coef_[i]), 6)
            for i in range(len(self.coef_))
        }
        return {
            "intercept": round(self.intercept_, 6),
            "weights": weights_dict
        }

    def get_params(self) -> Dict[str, Any]:
        return {"fit_intercept": self.fit_intercept, "rcond": self.rcond}

    def set_params(self, **params: Any) -> "LinearRegressionModel":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    @classmethod
    def load_from_json(cls, json_path: str) -> "LinearRegressionModel":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        model.fit_intercept = bool(data["hyperparameters"]["fit_intercept"])
        model.rcond = float(data["hyperparameters"]["rcond"])
        model.intercept_ = float(data["learned_parameters"]["intercept"])
        weights_dict = data["learned_parameters"]["weights"]
        model.feature_names_ = list(weights_dict.keys())
        model.coef_ = np.array(list(weights_dict.values()), dtype=np.float64)
        model.weights_ = np.array([model.intercept_] + list(weights_dict.values()), dtype=np.float64)
        model.n_features_in_ = len(model.feature_names_)
        model.is_fitted = True
        model.training_details_ = data.get("training_details", {})
        return model


class TreeNode:
    def __init__(
        self,
        node_id: int = 0,
        depth: int = 0,
        feature_idx: Optional[int] = None,
        feature_name: Optional[str] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[float] = None,
        n_samples: int = 0,
        mse: float = 0.0,
        variance_reduction: float = 0.0
    ) -> None:
        self.node_id = node_id
        self.depth = depth
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.mse = mse
        self.variance_reduction = variance_reduction

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressorModel:
    def __init__(self) -> None:
        self.max_depth: Optional[int] = None
        self.min_samples_leaf: Optional[int] = None
        self.min_samples_split: Optional[int] = None

        self.root: Optional[TreeNode] = None
        self.n_features_in_: int = 0
        self.feature_names_: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self._node_counter: int = 0

    def build_model(
        self,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int
    ) -> "DecisionTreeRegressorModel":
        if max_depth < 1:
            raise ValueError("max_depth phải >= 1.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf phải >= 1.")
        if min_samples_split < 2 * min_samples_leaf:
            min_samples_split = 2 * min_samples_leaf

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        self.root = None
        self.feature_importances_ = None
        self.is_fitted = False
        self._node_counter = 0
        return self

    def _to_numpy(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X_arr = np.array(X, dtype=np.float64)
            if not self.feature_names_:
                self.feature_names_ = [f"x_{i}" for i in range(X_arr.shape[1])]
        else:
            raise TypeError("X phải là numpy.ndarray hoặc pandas.DataFrame.")

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float64).ravel()
            elif isinstance(y, np.ndarray):
                y_arr = np.array(y, dtype=np.float64).ravel()
            else:
                raise TypeError("y phải là numpy.ndarray, pandas.Series hoặc DataFrame.")

            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError("Số dòng của X và y không khớp nhau.")

        return X_arr, y_arr

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, 0.0

        total_sum = np.sum(y)
        total_sq_sum = np.sum(y ** 2)
        parent_var = (total_sq_sum / n_samples) - ((total_sum / n_samples) ** 2)

        if parent_var <= 1e-9:
            return None, None, 0.0

        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        best_variance_reduction: float = 0.0
        best_score: float = -np.inf
        base_score = (total_sum ** 2) / n_samples

        for feat in range(n_features):
            x_col = X[:, feat]
            sort_order = np.argsort(x_col)
            x_sorted = x_col[sort_order]
            y_sorted = y[sort_order]

            split_candidates = x_sorted[:-1] < x_sorted[1:]
            split_candidates[:self.min_samples_leaf - 1] = False
            split_candidates[n_samples - self.min_samples_leaf:] = False

            if not np.any(split_candidates):
                continue

            cumsum_y = np.cumsum(y_sorted)
            k = np.arange(1, n_samples)

            valid_k = k[split_candidates]
            sum_left = cumsum_y[:-1][split_candidates]
            sum_right = total_sum - sum_left

            n_left = valid_k
            n_right = n_samples - valid_k

            scores = (sum_left ** 2 / n_left) + (sum_right ** 2 / n_right)
            max_idx = np.argmax(scores)
            max_score = scores[max_idx]

            if max_score > best_score:
                best_score = max_score
                best_split_k = valid_k[max_idx]
                best_feat = feat
                best_thresh = float((x_sorted[best_split_k - 1] + x_sorted[best_split_k]) / 2.0)
                best_variance_reduction = float((max_score - base_score) / n_samples)

        return best_feat, best_thresh, best_variance_reduction

    def _build_tree(self, X: np.ndarray, y: np.ndarray, current_depth: int, n_total_train: int) -> TreeNode:
        node_id = self._node_counter
        self._node_counter += 1

        n_samples = len(y)
        node_mean = float(np.mean(y)) if n_samples > 0 else 0.0
        node_mse = float(np.mean((y - node_mean) ** 2)) if n_samples > 0 else 0.0

        if (
            current_depth >= self.max_depth
            or n_samples < self.min_samples_split
            or node_mse < 1e-7
        ):
            return TreeNode(
                node_id=node_id,
                depth=current_depth,
                value=node_mean,
                n_samples=n_samples,
                mse=node_mse
            )

        best_feat, best_thresh, gain = self._find_best_split(X, y)
        if best_feat is None or best_thresh is None or gain <= 0.0:
            return TreeNode(
                node_id=node_id,
                depth=current_depth,
                value=node_mean,
                n_samples=n_samples,
                mse=node_mse
            )

        if self.feature_importances_ is not None and n_total_train > 0:
            self.feature_importances_[best_feat] += gain * (n_samples / n_total_train)

        feat_name = self.feature_names_[best_feat] if best_feat < len(self.feature_names_) else f"x_{best_feat}"

        left_mask = X[:, best_feat] <= best_thresh
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        left_child = self._build_tree(X_left, y_left, current_depth + 1, n_total_train)
        right_child = self._build_tree(X_right, y_right, current_depth + 1, n_total_train)

        return TreeNode(
            node_id=node_id,
            depth=current_depth,
            feature_idx=best_feat,
            feature_name=feat_name,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=None,
            n_samples=n_samples,
            mse=node_mse,
            variance_reduction=gain
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> "DecisionTreeRegressorModel":
        if self.max_depth is None or self.min_samples_leaf is None or self.min_samples_split is None:
            raise RuntimeError("Vui lòng gọi build_model(max_depth, min_samples_leaf, min_samples_split) trước khi fit!")

        X_mat, y_vec = self._to_numpy(X, y)
        n_samples, n_features = X_mat.shape
        self.n_features_in_ = n_features
        self._node_counter = 0

        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        self.root = self._build_tree(X_mat, y_vec, current_depth=0, n_total_train=n_samples)

        sum_imp = np.sum(self.feature_importances_)
        if sum_imp > 0:
            self.feature_importances_ = self.feature_importances_ / sum_imp

        self.is_fitted = True
        return self

    def _predict_single(self, node: TreeNode, x: np.ndarray) -> float:
        if node.is_leaf:
            return float(node.value)

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(node.left, x)
        else:
            return self._predict_single(node.right, x)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted or self.root is None:
            raise RuntimeError("Cây chưa được fit! Vui lòng gọi fit(X, y) trước.")

        X_mat, _ = self._to_numpy(X)
        if X_mat.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Kích thước đặc trưng không khớp: Cần {self.n_features_in_}, nhận được {X_mat.shape[1]}."
            )

        predictions = [self._predict_single(self.root, row) for row in X_mat]
        return np.array(predictions, dtype=np.float64)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Dict[str, float]:
        _, y_true = self._to_numpy(X, y)
        y_pred = self.predict(X)

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))) * 100.0

        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE(%)": round(mape, 4),
            "R2": round(r2, 4)
        }

    def get_nodes_list(self) -> List[Dict[str, Any]]:
        if not self.is_fitted or self.root is None:
            raise RuntimeError("Cây chưa được fit!")

        nodes: List[Dict[str, Any]] = []

        def _traverse(
            node: Optional[TreeNode],
            decision_path: str = "Root",
            parent_branch_condition: Optional[str] = None
        ) -> None:
            if node is None:
                return

            node_dict: Dict[str, Any] = {
                "node_id": node.node_id,
                "depth": node.depth,
                "node_type": "Leaf" if node.is_leaf else "Internal",
                "n_samples": node.n_samples,
                "mse": round(node.mse, 4)
            }

            if node.is_leaf:
                node_dict["predicted_value"] = round(float(node.value), 4)
                if parent_branch_condition is not None:
                    node_dict["branch_condition"] = parent_branch_condition
                node_dict["decision_path"] = decision_path
            else:
                threshold_val = round(float(node.threshold), 4)
                left_cond = f"{node.feature_name} <= {threshold_val}"
                right_cond = f"{node.feature_name} > {threshold_val}"

                node_dict["split_feature"] = node.feature_name
                node_dict["split_threshold"] = threshold_val
                node_dict["split_condition"] = left_cond
                node_dict["left_branch_condition"] = left_cond
                node_dict["right_branch_condition"] = right_cond
                node_dict["variance_reduction"] = round(float(node.variance_reduction), 4)
                node_dict["left_child_id"] = node.left.node_id if node.left else None
                node_dict["right_child_id"] = node.right.node_id if node.right else None
                if parent_branch_condition is not None:
                    node_dict["branch_condition"] = parent_branch_condition
                node_dict["decision_path"] = decision_path

            nodes.append(node_dict)

            if node.left:
                left_path = left_cond if decision_path == "Root" else f"{decision_path} and {left_cond}"
                _traverse(node.left, left_path, left_cond)
            if node.right:
                right_path = right_cond if decision_path == "Root" else f"{decision_path} and {right_cond}"
                _traverse(node.right, right_path, right_cond)

        _traverse(self.root)
        nodes.sort(key=lambda x: x["node_id"])
        return nodes

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or self.feature_importances_ is None:
            raise RuntimeError("Cây chưa được fit!")
        return {
            self.feature_names_[i]: round(float(self.feature_importances_[i] * 100.0), 2)
            for i in range(len(self.feature_importances_))
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split
        }

    def set_params(self, **params: Any) -> "DecisionTreeRegressorModel":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    @classmethod
    def load_from_json(cls, json_path: str) -> "DecisionTreeRegressorModel":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        params = data["hyperparameters"]
        model.build_model(
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params["min_samples_split"])
        )
        nodes_list = data["tree_structure"]["nodes"]
        feature_names = list(data.get("feature_importances", {}).keys())
        model.feature_names_ = feature_names
        model.n_features_in_ = len(feature_names)
        importances_dict = data.get("feature_importances", {})
        if importances_dict:
            model.feature_importances_ = np.array(
                [importances_dict.get(fn, 0.0) / 100.0 for fn in feature_names],
                dtype=np.float64
            )

        node_map: Dict[int, TreeNode] = {}
        for nd in nodes_list:
            nid = int(nd["node_id"])
            depth = int(nd["depth"])
            n_samples = int(nd["n_samples"])
            mse = float(nd["mse"])
            if nd["node_type"] == "Leaf":
                val = float(nd["predicted_value"])
                node_map[nid] = TreeNode(
                    node_id=nid, depth=depth, value=val, n_samples=n_samples, mse=mse
                )
            else:
                feat = str(nd["split_feature"])
                f_idx = feature_names.index(feat) if feat in feature_names else 0
                thresh = float(nd["split_threshold"])
                vr = float(nd.get("variance_reduction", 0.0))
                node_map[nid] = TreeNode(
                    node_id=nid, depth=depth, feature_idx=f_idx, feature_name=feat,
                    threshold=thresh, n_samples=n_samples, mse=mse, variance_reduction=vr
                )

        for nd in nodes_list:
            if nd["node_type"] == "Internal":
                nid = int(nd["node_id"])
                left_id = nd.get("left_child_id")
                right_id = nd.get("right_child_id")
                if left_id is not None:
                    node_map[nid].left = node_map.get(left_id)
                if right_id is not None:
                    node_map[nid].right = node_map.get(right_id)

        model.root = node_map.get(0)
        model.is_fitted = True
        return model

    def predict_single_with_path(self, x: np.ndarray) -> Dict[str, Any]:
        if not self.is_fitted or self.root is None:
            raise RuntimeError("Cây chưa được fit!")
        curr: Optional[TreeNode] = self.root
        path_conditions: List[str] = []
        while curr is not None and not curr.is_leaf:
            val = x[curr.feature_idx]
            thresh = curr.threshold if curr.threshold is not None else 0.0
            if val <= thresh:
                cond = f"{curr.feature_name} <= {round(float(thresh), 4)}"
                path_conditions.append(cond)
                curr = curr.left
            else:
                cond = f"{curr.feature_name} > {round(float(thresh), 4)}"
                path_conditions.append(cond)
                curr = curr.right

        pred = float(curr.value) if (curr is not None and curr.value is not None) else 0.0
        return {
            "predicted_value": round(pred, 4),
            "leaf_node_id": curr.node_id if curr else -1,
            "leaf_samples": curr.n_samples if curr else 0,
            "leaf_mse": round(curr.mse, 4) if curr else 0.0,
            "decision_path": " and ".join(path_conditions) if path_conditions else "Root"
        }


LinearRegression = LinearRegressionModel
DecisionTreeRegressor = DecisionTreeRegressorModel
