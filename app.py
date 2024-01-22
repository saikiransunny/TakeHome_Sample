import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError, validator
from typing import List
import pickle

model_features = ['easements', 'lotarea', 'bldgarea', 'resarea', 'officearea',
       'retailarea', 'garagearea', 'strgearea', 'factryarea', 'numbldgs',
       'numfloors', 'unitstotal', 'lotfront', 'lotdepth', 'bldgfront',
       'bldgdepth', 'assessland', 'yearbuilt', 'yearalter1', 'yearalter2',
       'builtfar', 'tract2010', 'xcoord', 'ycoord', 'borough', 'splitzone',
       'irrlotcode']

app = FastAPI()

# Load the pickled model
with open('model_artifacts/model.pickle', 'rb') as file:
    model = pickle.load(file)



class Item(BaseModel):
    '''
    The above class defines a data model for an item with various attributes, and includes validators to
    ensure that certain attributes have valid values.
    '''
    easements: float
    lotarea: float
    bldgarea: float
    resarea: float
    officearea: float
    retailarea: float
    garagearea: float
    strgearea: float
    factryarea: float
    numbldgs: float
    numfloors: float
    unitstotal: float
    lotfront: float
    lotdepth: float
    bldgfront: float
    bldgdepth: float
    assessland: float
    yearbuilt: float
    yearalter1: float
    yearalter2: float
    builtfar: float
    tract2010: float
    xcoord: float
    ycoord: float
    borough: str
    splitzone: str
    irrlotcode: str

    @validator("easements", "lotarea", "bldgarea", "resarea", "officearea","retailarea","garagearea",
               "strgearea","factryarea","numbldgs","numfloors","unitstotal","lotfront","lotdepth","bldgfront","bldgdepth","assessland","yearbuilt",
               "yearalter1","yearalter2","builtfar","tract2010","xcoord","ycoord",pre=True, each_item=True)
    def check_positive_float(cls, value):
        """
        The function `check_positive_float` checks if a value is a positive float or integer and raises a
        ValueError if it is not.
        
        :return: The value is being returned.
        """
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Numeric fields cannot be negative.")
        return value
    
    @validator("borough",pre=True, each_item=True)
    def check_borough_validity(cls, value):
        """
        The function `check_borough_validity` checks if a given value is a valid borough code.
        
        :return: the value of the "value" parameter if it is a valid borough value.
        """
        valid_values = ['BX', 'QN', 'BK', 'MN', 'SI']
        if value not in valid_values:
            raise ValueError("borough should be one of - 'BX', 'QN', 'BK', 'MN', 'SI'")
        return value
    
    @validator("splitzone",pre=True, each_item=True)
    def check_splitzone_validity(cls, value):
        """
        The function `check_splitzone_validity` checks if a given value is valid for the `splitzone`
        attribute, raising a `ValueError` if it is not.
        
        :return: The value of the variable "value" is being returned.
        """
        valid_values = ['N', 'Y']
        if value not in valid_values:
            raise ValueError("splitzone should be one of - 'N', 'Y'")
        return value
    
    @validator("irrlotcode",pre=True, each_item=True)
    def check_irrlotcode_validity(cls, value):
        """
        The function `check_irrlotcode_validity` checks if a given value is valid by comparing it to a list
        of valid values and raises a ValueError if it is not.

        :return: The value of the variable "value" is being returned.
        """
        valid_values = ['N', 'Y']
        if value not in valid_values:
            raise ValueError("irrlotcode should be one of - 'N', 'Y'")
        return value

class Prediction(BaseModel):
    '''
    The `Prediction` class is a subclass of `BaseModel` and has a single attribute `prediction` of type
    string.
    '''
    prediction: str

@app.post("/predict/", response_model=Prediction, summary="Make predictions", description="Predict if a property could be a potential office space")
async def predict(item: Item):
    """
    The `predict` function takes an `Item` object as input, extracts its features, creates a DataFrame,
    and uses a pre-trained model to make a prediction based on those features.
    
    :param item: The `item` parameter is an instance of the `Item` class. It contains various attributes
    that represent different features of an item. These attributes include:
    :type item: Item
    :return: a dictionary with a key "prediction" and the predicted value as the corresponding value.
    """
    features = [item.easements, item.lotarea, item.bldgarea, item.resarea, item.officearea, item.retailarea, item.garagearea, 
                 item.strgearea,item.factryarea,item.numbldgs,item.numfloors,item.unitstotal,item.lotfront,item.lotdepth,item.bldgfront,
                 item.bldgdepth,item.assessland,item.yearbuilt, item.yearalter1,item.yearalter2,item.builtfar,item.tract2010,item.xcoord,item.ycoord,
                 item.borough,item.splitzone,item.irrlotcode]
    df = pd.DataFrame([features])
    df.columns = model_features
    prediction = str(model.predict(df)[0])
    return {"prediction" : prediction}



# Add a simple route for the root path
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI ML service!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)