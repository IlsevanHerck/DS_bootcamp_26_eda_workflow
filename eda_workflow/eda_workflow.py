import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


def _get_date_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are native datetimes or ISO date strings."""
    date_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

    iso_date_pattern = r"^\d{4}-\d{2}-\d{2}"
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col in date_cols:
            continue
        series = df[col].dropna().astype(str)
        if series.empty or not series.str.match(iso_date_pattern).all():
            continue
        parsed = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
        if parsed.notna().all():
            date_cols.append(col)

    return date_cols


def _normalize_column_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def _get_agg_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return [
        col for col in numeric_cols
        if "_per_" not in f"_{_normalize_column_name(col)}_"
        and "_pr_" not in f"_{_normalize_column_name(col)}_"
    ]


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }

    def validate_data_integrity_node(state: EDAState):
        """Validate row integrity, logical consistency, and statistical outliers."""
        logger.info("Validating data integrity")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        total_rows = len(df)
        agg_numeric_cols = _get_agg_numeric_columns(df)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        duplicate_row_count = int(df.duplicated().sum())
        integrity = {
            "duplicate_rows": {
                "count": duplicate_row_count,
                "pct": round(duplicate_row_count / total_rows * 100, 2) if total_rows > 0 else 0,
            },
        }

        key_cols = [
            col for col in df.columns
            if _normalize_column_name(col).endswith("_id")
            or _normalize_column_name(col) == "id"
        ]
        duplicate_keys = {}
        for col in key_cols:
            duplicate_count = int(df[col].duplicated().sum())
            if duplicate_count > 0:
                duplicate_keys[col] = {
                    "count": duplicate_count,
                    "pct": round(duplicate_count / total_rows * 100, 2) if total_rows > 0 else 0,
                }
        integrity["duplicate_keys"] = duplicate_keys

        qty_cols = [
            col for col in numeric_cols
            if any(token in _normalize_column_name(col) for token in ("quantity", "qty", "count"))
        ]
        rate_cols = [
            col for col in numeric_cols
            if "_per_" in f"_{_normalize_column_name(col)}_"
            or "_pr_" in f"_{_normalize_column_name(col)}_"
        ]
        total_cols = [
            col for col in agg_numeric_cols
            if any(token in _normalize_column_name(col) for token in ("total", "spent", "amount", "sum"))
        ]

        consistency_violations = {}
        if qty_cols and rate_cols and total_cols:
            qty_col = qty_cols[0]
            rate_col = rate_cols[0]
            total_col = total_cols[0]
            expected_total = df[qty_col] * df[rate_col]
            violation_mask = (
                expected_total.notna()
                & df[total_col].notna()
                & ((expected_total - df[total_col]).abs() > 1e-6)
            )
            violation_count = int(violation_mask.sum())
            if violation_count > 0:
                consistency_violations["quantity_rate_total_mismatch"] = {
                    "quantity_column": qty_col,
                    "rate_column": rate_col,
                    "total_column": total_col,
                    "count": violation_count,
                    "pct": round(violation_count / total_rows * 100, 2) if total_rows > 0 else 0,
                }
        integrity["consistency_violations"] = consistency_violations

        non_negative_tokens = ("quantity", "qty", "count", "price", "total", "spent", "amount")
        invalid_ranges = {}
        for col in numeric_cols:
            if not any(token in _normalize_column_name(col) for token in non_negative_tokens):
                continue
            negative_count = int((df[col] < 0).sum())
            if negative_count > 0:
                invalid_ranges[col] = {
                    "negative_count": negative_count,
                    "pct": round(negative_count / total_rows * 100, 2) if total_rows > 0 else 0,
                }
        integrity["invalid_ranges"] = invalid_ranges

        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]
        integrity["constant_columns"] = constant_columns

        statistical_outliers = {}
        for col in agg_numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                statistical_outliers[col] = {
                    "count": outlier_count,
                    "pct": round(outlier_count / len(series) * 100, 2),
                    "lower_bound": round(float(lower_bound), 3),
                    "upper_bound": round(float(upper_bound), 3),
                }
        integrity["statistical_outliers"] = statistical_outliers

        results["validate_data_integrity"] = integrity

        return {
            "current_step": "validate_data_integrity",
            "results": results,
        }
    
    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns.
        
        TODO: Implement this analysis tool.
        
        See profile_dataset_node and analyze_missingness_node for reference.
        Store your results in results["compute_aggregates"] and return
        {"current_step": "compute_aggregates", "results": results}.
        """
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns
        agg_numeric_cols = [
            col for col in numeric_cols
            if "_per_" not in f"_{col.lower().replace(' ', '_')}_"
            and "_pr_" not in f"_{col.lower().replace(' ', '_')}_"
        ]
        numeric_sums = (
            df[agg_numeric_cols].sum().to_dict() if len(agg_numeric_cols) > 0 else {}
        )

        df_aggregates = {}
        df_aggregates["numeric_sums"] = numeric_sums

        date_cols = _get_date_columns(df)

        if len(date_cols) == 1 and len(agg_numeric_cols) > 0:
            date_col = date_cols[0]
            dates = pd.to_datetime(df[date_col], format="ISO8601", errors="coerce")
            yearly_sums = (
                df.assign(_year=dates.dt.year)
                .groupby("_year", sort=True)[agg_numeric_cols]
                .sum()
                .to_dict(orient="index")
            )
            quarterly_sums = (
                df.assign(_quarter=dates.dt.to_period("Q").astype(str))
                .groupby("_quarter", sort=True)[agg_numeric_cols]
                .sum()
                .to_dict(orient="index")
            )
            monthly_sums = (
                df.assign(_month=dates.dt.to_period("M").astype(str))
                .groupby("_month", sort=True)[agg_numeric_cols]
                .sum()
                .to_dict(orient="index")
            )
            df_aggregates["yearly_numeric_sums"] = yearly_sums
            df_aggregates["quarterly_numeric_sums"] = quarterly_sums
            df_aggregates["monthly_numeric_sums"] = monthly_sums
        
        results["compute_aggregates"] = df_aggregates
        
        return {
            "current_step": "compute_aggregates",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables.
        
        TODO: Implement this analysis tool.
        
        See profile_dataset_node and analyze_missingness_node for reference.
        Store your results in results["analyze_relationships"] and return
        {"current_step": "analyze_relationships", "results": results}.
        """
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        agg_numeric_cols = [
            col for col in numeric_cols
            if "_per_" not in f"_{col.lower().replace(' ', '_')}_"
            and "_pr_" not in f"_{col.lower().replace(' ', '_')}_"
        ]

        date_cols = _get_date_columns(df)

        relationship_categorical_cols = [
            col for col in categorical_cols
            if col not in date_cols and df[col].nunique() <= 20
        ]

        relationships = {}

        if len(agg_numeric_cols) >= 2:
            correlation_matrix = df[agg_numeric_cols].corr().round(3)
            numeric_correlations = correlation_matrix.to_dict()
            relationships["numeric_correlations"] = numeric_correlations

            strong_correlations = []
            for i, col_a in enumerate(agg_numeric_cols):
                for col_b in agg_numeric_cols[i + 1:]:
                    coefficient = correlation_matrix.loc[col_a, col_b]
                    if pd.notna(coefficient) and abs(coefficient) >= 0.5:
                        strong_correlations.append({
                            "column_a": col_a,
                            "column_b": col_b,
                            "correlation": float(coefficient),
                        })
            relationships["strong_correlations"] = strong_correlations

        if relationship_categorical_cols and agg_numeric_cols:
            categorical_numeric_means = {}
            for cat_col in relationship_categorical_cols:
                group_means = (
                    df.groupby(cat_col, sort=False)[agg_numeric_cols]
                    .mean()
                    .round(3)
                    .to_dict(orient="index")
                )
                categorical_numeric_means[cat_col] = group_means
            relationships["categorical_numeric_means"] = categorical_numeric_means

        results["analyze_relationships"] = relationships

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }

    def analyze_timeseries_node(state: EDAState):
        """Analyze timeseries if available.
        
        See profile_dataset_node and compute_aggregates_node for reference.
        Store your results in results["analyze_timeseries"] and return
        {"current_step": "analyze_timeseries", "results": results}.
        """
        logger.info("Analyzing timeseries")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns
        agg_numeric_cols = [
            col for col in numeric_cols
            if "_per_" not in f"_{col.lower().replace(' ', '_')}_"
            and "_pr_" not in f"_{col.lower().replace(' ', '_')}_"
        ]

        date_cols = _get_date_columns(df)

        timeseries = {}

        if len(date_cols) == 1:
            date_col = date_cols[0]
            dates = pd.to_datetime(df[date_col], format="ISO8601", errors="coerce")
            unique_dates = dates.drop_duplicates().sort_values()

            timeseries["date_column"] = date_col
            timeseries["date_range"] = {
                "start": str(dates.min().date()),
                "end": str(dates.max().date()),
                "days": int((dates.max() - dates.min()).days),
            }
            timeseries["unique_dates"] = int(unique_dates.shape[0])

            if len(unique_dates) >= 2:
                median_gap_days = unique_dates.diff().dropna().dt.days.median()
                timeseries["median_gap_days"] = (
                    float(median_gap_days) if pd.notna(median_gap_days) else None
                )

            monthly_groups = dates.dt.to_period("M").astype(str)
            timeseries["monthly_record_count"] = (
                df.groupby(monthly_groups, sort=True).size().to_dict()
            )

            if len(agg_numeric_cols) > 0:
                monthly_totals = (
                    df.groupby(monthly_groups, sort=True)[agg_numeric_cols]
                    .sum()
                )
                monthly_pct_change = (
                    monthly_totals.pct_change()
                    .round(3)
                    .dropna(how="all")
                )
                timeseries["monthly_pct_change"] = (
                    monthly_pct_change.to_dict(orient="index")
                    if not monthly_pct_change.empty else {}
                )

                strong_monthly_trends = []
                if not monthly_pct_change.empty and len(monthly_pct_change) >= 2:
                    for col in agg_numeric_cols:
                        if col not in monthly_pct_change.columns:
                            continue
                        col_changes = monthly_pct_change[col].dropna()
                        if len(col_changes) < 2:
                            continue

                        for month, change in col_changes.items():
                            other_changes = col_changes.drop(month)
                            if other_changes.empty:
                                continue
                            rest_mean = other_changes.mean()
                            rest_std = other_changes.std()

                            if pd.isna(rest_std) or rest_std == 0:
                                is_strong = (
                                    abs(change - rest_mean) >= 0.2
                                    and abs(change - rest_mean) > abs(rest_mean)
                                )
                            else:
                                z_vs_rest = (change - rest_mean) / rest_std
                                is_strong = abs(z_vs_rest) >= 1.5

                            if is_strong:
                                direction = "upward" if change > rest_mean else "downward"
                                strong_monthly_trends.append({
                                    "month": month,
                                    "column": col,
                                    "pct_change": float(change),
                                    "rest_mean_pct_change": float(rest_mean),
                                    "direction": direction,
                                })

                timeseries["strong_monthly_trends"] = strong_monthly_trends

        results["analyze_timeseries"] = timeseries

        return {
            "current_step": "analyze_timeseries",
            "results": results,
        }
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("validate_data_integrity", validate_data_integrity_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_5", extract_observations_node)
    workflow.add_node("analyze_timeseries", analyze_timeseries_node)
    workflow.add_node("extract_observations_6", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "validate_data_integrity")
    workflow.add_edge("validate_data_integrity", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_5")
    workflow.add_edge("extract_observations_5", "analyze_timeseries")
    workflow.add_edge("analyze_timeseries", "extract_observations_6")
    workflow.add_edge("extract_observations_6", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
