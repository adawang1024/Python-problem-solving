# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name:
# Collaborators:
# Time:

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

        # NOTE: TO BE IMPLEMENTED IN PART 4B OF THE PSET
        raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError




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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError



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
    raise NotImplementedError



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
    raise NotImplementedError

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
    raise NotImplementedError


if __name__ == '__main__':
    pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE


    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE


    ##################################################################################
    # Problem 5B: INCREASING TRENDS


    ##################################################################################
    # Problem 5C: DECREASING TRENDS


    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.

    ##################################################################################
    # Problem 6B: PREDICTING


    ##################################################################################
    # Problem 7: KMEANS CLUSTERING (Checkoff Question Only)


    ####################################################################################