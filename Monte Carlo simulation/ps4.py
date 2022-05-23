from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import PIL, PIL.Image

import scipy.stats as st
# from scipy.interpolate import interp1d

class floodmapper:
    """
    Create a floodmapper instance, for a map in a specified bounding box and a particular mass gis dbf file.
    Parameters are:
        tl - (lat,lon) pair specifying the upper left corner of the map data to load

        br - (lat,lon) pair specifying the bottom right corner of the map data to load 

        z - Zoom factor of the map data to load
        
        dbf_path - Path to the dbf file listing properties to load
        
        load - Boolean specifying whether or not to download the map data from mapbox.  If false, mapbox files must 
        have been previously loaded.  If using this as a 6.0001 problem set, you should have received pre-rendered map tiles
        such that it is OK to pass false to this when using the specific parameters in the __main__ code block below.
    """
    def __init__(self, tl, br, z, dbf_path, load):
        self.mtl = maptileloader.maptileloader(tl, br, z)
        dbf = massdbfloader.massdbfloader(dbf_path)
        if (load):
            self.mtl.download_tiles()
        self.pts = dbf.get_points_in_box(tl,br)
        self.ul, self.br = self.mtl.get_tile_extents()
        self.elevs = self.mtl.get_elevation_array()

    """
    Return a rendering as a PIL image of the map where properties below elev are highlighted
    """
    def get_image_for_elev(self, elev):
        fnt = ImageFont.truetype("Arial.ttf", 80)
        im = self.mtl.get_satellite_image()
        draw = ImageDraw.Draw(im)
        draw.text((10,10), f"{elev} meters", font=fnt, fill=(255,255,255,128))
        for name, lat, lon, val in self.pts:
            # print(name)
            x = int((((lon - self.ul[0]) / (self.br[0] - self.ul[0]))) * self.elevs.shape[1])
            y = int((((lat - self.ul[1]) / (self.br[1] - self.ul[1]))) * self.elevs.shape[0])
            # print(x,y)
            # print(e[x,y])
            el = int(self.elevs[y,x]*15)
            #print(e[y,x])
            c = f"rgb(0,{el},200)"
            if (self.elevs[y,x] < elev):
                c = f"rgb(255,0,0)"
            draw.ellipse((x-3,y-3,x+3,y+3), PIL.ImageColor.getrgb(c))
        return im

    """
    Return an array of (property name, lat, lon, elevation (m), value (USD)) tuples where properties
    are below the specified elev.
    """
    def get_properties_below_elev(self, elev):
        out = []
        for name, lat, lon, val in self.pts:
            x = int((((lon - self.ul[0]) / (self.br[0] - self.ul[0]))) * self.elevs.shape[1])
            y = int((((lat - self.ul[1]) / (self.br[1] - self.ul[1]))) * self.elevs.shape[0])
            if (self.elevs[y,x] < elev):
                out.append((name,lat,lon, self.elevs[y,x], val))

        return out



#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper - mean) / st.norm.ppf(.975)


def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)


def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())


###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    raise NotImplementedError


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    raise NotImplementedError
    
    
def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    raise NotImplementedError
    

##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    raise NotImplementedError


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    raise NotImplementedError


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    raise NotImplementedError



def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    raise NotImplementedError


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    raise NotImplementedError



if __name__ == '__main__':
    
    # Comment out the 'pass' statement below to run the lines below it
    pass 

    import maptileloader
    import massdbfloader

    # # Uncomment the following lines to plot generate plots
    # data = predicted_sea_level_rise()
    # water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    # water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    # plot_mc_simulation(data)
    # plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
    
    # # Uncomment the following lines to visualize sea level rise over a map of Boston
    # tl = (42.3586798 +.04, - 71.1000466 - .065)
    # br = (42.3586798 -.02, - 71.1000466 + .065)
    # dbf = 'cambridge_2021.dbf'
    # fm = floodmapper(tl,br,14,dbf,False)

    # print("Getting properties below 5m")
    # pts = fm.get_properties_below_elev(5.0)
    # print(f"First one: {pts[0]}")
    
    # print("The next few steps may take a few seconds each.")

    # fig, ax = plt.subplots(figsize=(12,10), dpi=144)
    
    # ims=[]
    # print("Generating image frames for different elevations")
    # for el_cutoff in np.arange(0,15,.5):
    #     # print(el_cutoff)
    #     im = fm.get_image_for_elev(el_cutoff)
    #     im_plt = ax.imshow(im, animated=True)
    #     if el_cutoff == 0:
    #         ax.imshow(im)  # show an initial one first

    #     ims.append([im_plt])

    # print("Building animation")
    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
    #                                 repeat_delay=1000)
    # print("Saving animation to animation.gif")
    # ani.save('animation.gif', fps=30)

    # print("Displaying Image")
    # plt.show()
