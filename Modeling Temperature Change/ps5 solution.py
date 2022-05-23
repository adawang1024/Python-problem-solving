# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Yiduo Wang
# Collaborators:None
# Time: 12:00

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re
from sklearn.cluster import KMeans

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

# KMeans class not required until Problem 7
class KMeansClustering(KMeans):

    def __init__(self, data, k):
        super().__init__(n_clusters=k, random_state=0)
        self.fit(data)
        self.labels = self.predict(data)

    def get_centroids(self):
        'return np array of shape (n_clusters, n_features) representing the cluster centers'
        return self.cluster_centers_

    def get_labels(self):
        'Predict the closest cluster each sample in data belongs to. returns an np array of shape (samples,)'
        return self.labels

    def total_inertia(self):
        'returns the total inertia of all clusters, rounded to 4 decimal points'
        return round(self.inertia_, 4)



class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        all_annual = []
        # NOTE: TO BE IMPLEMENTED IN PART 4B OF THE PSET
        total_temps = 0
        for year in years:
            for city in cities:
                daily_temperatures = self.get_daily_temps(city, year)
                # get the mean of all year temperatures
                each_city_average = np.mean(daily_temperatures)
                # get all cities' total temperatures 
                total_temps += each_city_average
            # get the average temperature each year
            annual_temp_averages = total_temps/len(cities)
            all_annual.append(annual_temp_averages)
            # set total_temps back to 0 for each year
            total_temps = 0
            
        array = np.array(all_annual)
        return array
       
       
        
        
def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    # get the average of x and y
    x_average = np.mean(x)
    y_average = np.mean(y)
    # initialization
    sum_numerator = 0
    sum_denominator = 0
    for i in range(len(x)):
        # get numerator
        each_numerator = (x[i]-x_average)*(y[i]-y_average)
        sum_numerator += each_numerator
        # get denominator
        each_denominator = (x[i]-x_average)**2
        sum_denominator += each_denominator
        
    m = sum_numerator/sum_denominator
    b = y_average - m*x_average
        
    
    return (m,b)

def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    total_se = 0
    for i in range(len(x)):
        est_y = m*x[i]+b
        se = (y[i]-est_y)**2
        total_se += se
    return total_se




def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    all_models = []
    for degree in degrees:
        coefficient = np.polyfit(x,y,degree)
        all_models.append(coefficient)
    return all_models


def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    
    all_r2 = []
    for model in models:
        all_y = np.polyval(model,x)
        #  R-squared and standard error/slope should be rounded to 4 decimal places.
        each_r2 = round(r2_score(y, all_y),4)
        all_r2.append(each_r2)
        # plotting
        if display_graphs:
            degree = len(model)-1
            if degree == 1:
                ses = round(standard_error_over_slope(x, y, all_y, model),4)
               
            
            plt.scatter(x,y,c ='b')
            plt.plot(x,all_y,'r')
            plt.title(' Degree: ' + str(degree) + ' R-Square: ' + str(each_r2))
            # constant term is also included in the coefficient list
            degree = len(model)-1
            # include ses if slop = 1
            if degree == 1:
                plt.title(' Degree: ' + str(degree) + ' R-Square: ' + str(each_r2)+' SE/slope value: '+ str(ses))
            plt.xlabel("Years")
            plt.ylabel("Temperature in degrees Celsius")
            # plt.legend()
            plt.show()

    
    return all_r2

    


def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
# use a tolerance of 1e-8 to compare slope values (i.e. a float x and a float y are considered equal if abs(x - y) <= 1e-8abs(x−y)<=1e−8). If there are any ties, we resolve ties by returning the interval which occured first.
   # initialization
    extreme = 0
    ans_i = 0
    ans_j = 0
    
    for i in range(len(x)-length+1):
            j = i+length
            # get x and y
            x_array = x[i:j]
            y_array = y[i:j]
            m,b = linear_regression(x_array, y_array)
            if positive_slope:
                # positive slope
                if abs(m - extreme) > 1e-8 and m > extreme:
                    # update the values
                        extreme = m 
                        ans_i= i
                        ans_j = j
            if not positive_slope:
                # negative slope
                if m < extreme and abs(m-extreme) >1e-8:
                    # update the values
                        extreme = m
                        ans_i= i
                        ans_j = j
    # extreme would change if a proper interval is found; if not, return none
    if extreme == 0:
        return None
    
    return (ans_i,ans_j,extreme)
        


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    
    result = []
    for interval in range(2,len(x)+1):
    # when positive extreme exists
        if get_max_trend(x, y, interval,positive_slope = True):
            i1,j1,m1 = get_max_trend(x, y, interval,positive_slope = True)
            # when negative extreme exists as well
            if get_max_trend(x, y, interval,positive_slope = False):
                i2,j2,m2 = get_max_trend(x, y, interval,positive_slope = False)
                if (m1 - abs(m2))>1e-8:
                    result.append((i1,j1,m1))
                else:
                    result.append((i2,j2,m2))
            else:
                result.append((i1,j1,m1))

        else:
            # when only negative extreme exists
            if get_max_trend(x, y, interval,positive_slope = False):
                i2,j2,m2 = get_max_trend(x, y, interval,positive_slope = False)
                result.append((i2,j2,m2))
            #  none exists
            else:
                result.append((0, interval, None))
            
    return result
        
        

# Use the provided variables TRAIN_INTERVAL and TEST_INTERVAL to represent these ranges of years

def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    n = len(y)
    difference = y - estimated
    difference_sqr = difference**2
    sum_difference_sqr = np.sum(difference_sqr)
    rmse = (sum_difference_sqr/n)**0.5
    return rmse



def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    rmse = []
    for model in models:
        predict_y = np.polyval(model,x)
        # print(predict_y.shape)
        each_rmse = round(calculate_rmse(y, predict_y),4)
        rmse.append(each_rmse)
        # plotting
        if display_graphs:
            degree = len(model)-1
            plt.scatter(x,y,c ='b')
            plt.plot(x,predict_y,'r')
            plt.title(' Degree: ' + str(degree) + ' RMSE: ' + str(each_rmse))
            plt.xlabel("Years")
            plt.ylabel("Temperature in degrees Celsius")
            # plt.legend()
            plt.show()
            
    return rmse
    

def cluster_cities(cities, years, data, n_clusters):
    '''
    Clusters cities into n_clusters clusters using their average daily temperatures
    across all years in years. Generates a line plot with the average daily temperatures
    for each city. Each cluster of cities should have a different color for its
    respective plots.

    Args:
        cities: a list of the names of cities to include in the average
                daily temperature calculations
        years: a list of years to include in the average daily
                temperature calculations
        data: a Dataset instance
        n_clusters: an int representing the number of clusters to use for k-means

    Note that this part has no test cases, but you will be expected to show and explain
    your plots during your checkoff
    '''
    # convert the daily temperature data for each city in cities into a numpy array feature vector
    data = Dataset("data.csv")
    total_temp = np.empty((0,365))
    for city in cities:
        # print(total_temp.shape)
        sum_temp = np.full(365, float(0))
        for year in years:
            yearly_temp = data.get_daily_temps(city, year)[:365]
            # yearly_list = yearly_temp.tolist()
            sum_temp += yearly_temp
        # print(sum_temp.shape)
        average_temp = sum_temp/len(years)
        # y_vals = average_temp
        total_temp = np.append(total_temp,[average_temp],axis = 0)
    # cluster the feature vectors using the Clustering class
    clusters = KMeansClustering(total_temp, n_clusters)
    # initialize a list to hold labels
    labels = []
    C0 = []
    C1 = []
    C2 = []
    C3 = []
    x_vals = np.array([i for i in range(1,366)]) 
    for i in range(len(cities)):
        label = "C" + str(clusters.labels_[i])
        if label not in labels:
            # x-axis: days of the year, 1-365
            # y-axis: degrees celcius
            # each city feature vector should be colored according to the cluster it was associated with
            plt.plot(x_vals,total_temp[i],color = label,label = "Cluster" + str(clusters.labels_[i]))
            labels.append(label)
        else:
            plt.plot(x_vals,total_temp[i],color = label)
    
        # different clusters
        if clusters.labels_[i] == 0:
            C0.append(cities[i])
        if clusters.labels_[i] == 1:
            C1.append(cities[i])
        if clusters.labels_[i] == 2:
            C2.append(cities[i])
        if clusters.labels_[i] == 3:
            C3.append(cities[i])
         
    plt.xlabel("Days in a year")
    plt.ylabel("Temperature in degrees Celsius")
    plt.legend()
    plt.title("Cities Grouped Based on Climates")
        
    plt.show()
    
    
       
if __name__ == '__main__':
    pass
    ##################################################################################
    # # Problem 4A: DAILY TEMPERATURE
    # data = Dataset("data.csv")
    # x_coordinates = []
    # y_coordinates = []
    # for year in range(1961,2017):
    #     x_coordinates.append(year)
    #     each_y = data.get_temp_on_date("BOSTON", 12, 1, year)
    #     y_coordinates.append(each_y)
    # x = np.array(x_coordinates)
    # y = np.array(y_coordinates)
    # models = generate_polynomial_models(x, y, [1])
    # evaluate_models(x, y, models, display_graphs=True)
         
    ##################################################################################
    # # Problem 4B: ANNUAL TEMPERATURE
    # data = Dataset("data.csv")
    # x_coordinates = []
    # for year in range(1961,2017):
    #     x_coordinates.append(year)
    # x = np.array(x_coordinates)
    # y = data.calculate_annual_temp_averages(["BOSTON"], x_coordinates)
    # models = generate_polynomial_models(x, y, [1])
    # evaluate_models(x, y, models, display_graphs=True)

    ##################################################################################
    # # Problem 5B: INCREASING TRENDS
    # # Use get_max_trend to identify a window of 30 years
    # # demonstrates that the average annual temperature in Seattle is rising. 
    # # Plot the corresponding model with evaluate_models.
    # data = Dataset("data.csv")
    # x_coordinates = []
    # for year in range(1961,2017):
    #     x_coordinates.append(year)
    # x = np.array(x_coordinates)
    # y = data.calculate_annual_temp_averages(["SEATTLE"], x_coordinates)
    # i,j,m = get_max_trend(x, y, 30, positive_slope = True)
    # x_vals = x[i:j]
    # y_vals = y[i:j]
    # model = generate_polynomial_models(x_vals, y_vals, [1])
    # evaluate_models(x_vals, y_vals, model, display_graphs=True)
    
    ##################################################################################
    # # Problem 5C: DECREASING TRENDS
    # data = Dataset("data.csv")
    # x_coordinates = []
    # for year in range(1961,2017):
    #     x_coordinates.append(year)
    # x = np.array(x_coordinates)
    # y = data.calculate_annual_temp_averages(["SEATTLE"], x_coordinates)
    # i,j,m = get_max_trend(x, y, 15, positive_slope = False)
    # x_vals = x[i:j]
    # y_vals = y[i:j]
    # model = generate_polynomial_models(x_vals, y_vals, [1])
    # evaluate_models(x_vals, y_vals, model, display_graphs=True) 

    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.
    
    ##################################################################################
    # # Problem 6B: PREDICTING
    # # a training set of the national annual average temperature for the years in TRAIN_INTERVAL.
    # data = Dataset("data.csv") 
    # x_training = np.array(list(TRAIN_INTERVAL))
    # y_training = data.calculate_annual_temp_averages(CITIES, list(TRAIN_INTERVAL))
    # models = generate_polynomial_models(x_training, y_training, [2,10])
    # evaluate_models(x_training, y_training, models, display_graphs=True)

    # x_test = np.array(list(TEST_INTERVAL))
    # y_test = data.calculate_annual_temp_averages(CITIES, list(TEST_INTERVAL))
    # evaluate_rmse(x_test, y_test, models, display_graphs=True)

    ##################################################################################
    # # Problem 7: KMEANS CLUSTERING (Checkoff Question Only)
    # cities = CITIES
    # years = list(range(1961,2017))
    # data = Dataset("data.csv") 
    # n_clusters = 4

    ####################################################################################
