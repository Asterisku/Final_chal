from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import datetime
import json
import numpy as np
import sys
from pyspark.sql.functions import concat, lit

def main(sc, spark):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''

    dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
    dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
    OUTPUT_PREFIX = sys.argv[1]
    CAT_CODES = {'445210', '722515', '445299', '445120', '452210', '311811', '722410', '722511', '445220',
             '445292', '445110', '445291', '445230', '446191', '446110', '722513', '452311'}
    CAT_GROUP = {'452311': 0, '452210': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446191': 5, '446110': 5,
             '722515': 6, '311811': 6, '445299': 7, '445220': 7, '445292': 7, '445291': 7, '445230': 7, '445210': 7, '445110': 8}
    dfD = dfPlaces. \
    filter(dfPlaces.naics_code.isin(CAT_CODES)).select('placekey', 'naics_code')
    udfToGroup = F.udf(lambda x: CAT_GROUP[x])

    dfE = dfD.withColumn('group', udfToGroup('naics_code'))
    dfF = dfE.drop('naics_code').cache()
    def expandVisits(date_range_start, visits_by_day):
        from datetime import datetime, timedelta

        visits = visits_by_day.replace('[', '').replace(']', '').split(',')
        for i in range(7):
            date_time_object = datetime.strptime(date_range_start[0:10], "%Y-%m-%d")  + timedelta(days=i)
            month_day = date_time_object.strftime("%m-%d")
            year = date_time_object.strftime("%Y")
            yield(int(year), month_day, int(visits[i]))

    visitType = T.StructType([T.StructField('year', T.IntegerType()),
                          T.StructField('date', T.StringType()),
                          T.StructField('visits', T.IntegerType())])

    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

    dfH = dfPattern.join(dfF, 'placekey') \
        .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
        .select('group', 'expanded.*')
        
    def computeStats(group, visits):
        median = np.median(visits)
        stand_devi = np.std(visits)
        high = median + stand_devi
        low = median - stand_devi
        if low < 0:
            low = 0
        return (int(median),int(low),int(high))

    statsType = T.StructType([T.StructField('median', T.IntegerType()),
                          T.StructField('low', T.IntegerType()),
                          T.StructField('high', T.IntegerType())])

    udfComputeStats = F.udf(computeStats, statsType)

    dfI = dfH.groupBy('group', 'year', 'date') \
        .agg(F.collect_list('visits').alias('visits')) \
        .withColumn('stats', udfComputeStats('group', 'visits'))
    
    dfJ = dfI \
        .withColumn('date', concat(dfI.year, lit('-'), dfI.date)) \
        .select('group', 'year', 'date' ,'stats.*') \
        .sort(dfI.group, dfI.year, dfI.date) \
        .cache() 
    
    
    dfJ.filter(f'group=0') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/big_box_grocers',
               mode='overwrite', header=True)
    dfJ.filter(f'group=1') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/convenience_stores',
               mode='overwrite', header=True)
    dfJ.filter(f'group=2') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/drinking_places',
               mode='overwrite', header=True)
    dfJ.filter(f'group=3') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/full_service_restaurants',
               mode='overwrite', header=True)
    dfJ.filter(f'group=4') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/limited_service_restaurants',
               mode='overwrite', header=True)
    dfJ.filter(f'group=5') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/pharmacies_and_drug_stores',
               mode='overwrite', header=True)
    dfJ.filter(f'group=6') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/snack_and_retail_bakeries',
               mode='overwrite', header=True)
    dfJ.filter(f'group=7') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/specialty_food_stores',
               mode='overwrite', header=True)
    dfJ.filter(f'group=8') \
    .drop('group') \
    .coalesce(1) \
    .write.csv(f'{OUTPUT_PREFIX}/supermarkets_except_convenience_stores',
               mode='overwrite', header=True)



if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)