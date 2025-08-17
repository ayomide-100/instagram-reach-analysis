import numpy as np
import pandas as pd
import joblib

grid_search = joblib.load('../models/ml_models/best_imp_model.pkl')

imp_model = grid_search.best_estimator_

sample = {
    "CaptionLength": 450,
    "DayOfWeek": 2,
    "HashtagDensity": 0.12,
    "IsWeekend": 0,
    "Month": 7,
    "HourOfDay": 16,
    "NumOfHashtags": 12,
    "Content_Type_enc": 2,
    "Caption": (
        "We are thrilled to announce the expansion of our AI-driven analytics platform. "
        "This milestone comes after months of research, testing, and invaluable feedback "
        "from our early adopters. With this update, users can now track performance metrics, "
        "optimize workflows, and gain deep insights into customer behavior like never before. "
        "Our team worked tirelessly to ensure scalability, security, and accuracy, and this "
        "is just the beginning of many more improvements to come."
    ),
    "Hashtags": (
        "#ArtificialIntelligence #MachineLearning #DataScience #DeepLearning "
        "#BigData #AIinBusiness #TechInnovation #Automation #CloudComputing "
        "#FutureOfWork #AItools #DigitalTransformation"
    )
}



sample_df = pd.DataFrame([sample])

pred = imp_model.predict(sample_df)
print(pred[0])