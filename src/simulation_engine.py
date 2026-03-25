import pandas as pd
import joblib

MODEL_PATH = "models/thermal_comfort_model.pkl"

FEATURE_COLUMNS = [
    "Air temperature (C)",
    "Relative humidity (%)",
    "Air velocity (m/s)",
    "Radiant temperature (C)",
    "Clo",
    "Met"
]


class ThermalComfortDigitalTwin:
    """
    Digital Twin for indoor thermal comfort simulation.
    """

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, inputs: dict):
        """
        Predict comfort class for given indoor conditions.
        """
        df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
        prediction = self.model.predict(df)[0]
        return prediction

    def run_scenario(self, base_conditions: dict, changes: dict):
        """
        Simulate a scenario by modifying base conditions.
        """
        scenario = base_conditions.copy()
        scenario.update(changes)

        comfort = self.predict(scenario)
        return scenario, comfort
