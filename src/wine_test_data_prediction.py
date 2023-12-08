"""
Chaitanya Karnati
ucid - ck338
"""
"""
Spark application to run tuned model with testfile
"""
import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

def clean_data(df):
    # cleaning header 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

    

"""main function for application"""
if __name__ == "__main__":
    
    # Create spark application
    spark = SparkSession.builder \
        .appName('Mahidhar_cs643_wine_prediction') \
        .getOrCreate()

    # create spark context to report logging information related spark
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Load and parse the data file into an RDD of LabeledPoint.
    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if not("/" in input_path):
            input_path = "data/csv/" + input_path
        model_path="/code/data/model/testdata.model"
        print("----Input file for test data is---")
        print(input_path)
    else:
        current_dir = os.getcwd() 
        print("-----------------------")
        print(current_dir)
        input_path = os.path.join(current_dir,"Downloads\pySparkAWSWinePredictionApp-main\pySparkAWSWinePredictionApp-main\data\csv\testdata.csv")
        model_path= os.path.join(current_dir, "Downloads\pySparkAWSWinePredictionApp-main\pySparkAWSWinePredictionApp-main\data\model\testdata.model")

    # read csv file in DataFram 
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    df1 = clean_data(df)
    # Split the data into training and test sets (30% held out for testing)
    # removing column not adding much value to prediction
    # removed 'residual sugar','free sulfur dioxide',  'pH',
    all_features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'chlorides',
                        'total sulfur dioxide',
                        'density',
                        'sulphates',
                        'alcohol',
                    ]
    
   
   
    rf = PipelineModel.load(model_path)
    
    predictions = rf.transform(df1)
    print(predictions.show(5))
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(
                                            labelCol='label', 
                                            predictionCol='prediction', 
                                            metricName='accuracy')

    # printing accuracy of above model
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy of wine prediction model = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())
    sys.exit(0)
