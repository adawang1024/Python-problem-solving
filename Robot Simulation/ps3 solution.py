# -*- coding: utf-8 -*-
# Problem Set 3: Simulating robots
# Name:Yiduo Wang
# Collaborators (discussion):
# Time:10:00

import math
import random
import matplotlib

from ps3_visualize import *
import pylab

# === Provided class Position, do NOT change
class Position(object):
    """
    A Position represents a location in a two-dimensional room, where
    coordinates are given by floats (x, y).
    """
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_new_position(self, angle, speed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does NOT test whether the returned position fits inside the room.

        angle: float representing angle in degrees, 0 <= angle < 360
        speed: positive float representing speed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.get_x(), self.get_y()

        # Compute the change in position
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))

        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y

        return Position(new_x, new_y)

    def __str__(self):
        return "Position: " + str(math.floor(self.x)) + ", " + str(math.floor(self.y))

# === Problem 1
class Room(object):
    """
    A Room represents a rectangular region containing clean or dusty
    tiles.

    A room has a width and a height and contains (width * height) tiles. Each tile
    has some fixed amount of dust. The tile is considered clean only when the amount
    of dust on this tile is 0.
    """
    def __init__(self, width, height, dust_amount):
        """
        Initializes a rectangular room with the specified width, height, and
        dust_amount on each tile.

        width: an integer > 0
        height: an integer > 0
        dust_amount: an integer >= 0
        """
        self.width = width
        self.height = height
        self.dust_amount = dust_amount
        # create a dictionary that maps each tile(represent by a (x,y)tuple)) to the dust it has
        self.tile = {}
        for i in range(width):
            for j in range(height):
                self.tile[(i,j)] = self.dust_amount
 
    def get_dust_amount(self, w, h):
        """
        Return the amount of dust on the tile (w, h)

        Assumes that (w, h) represents a valid tile inside the room.

        w: an integer
        h: an integer

        Returns: a float
        """
       # the value of dictionary is dust
        return self.tile[(w,h)]

    def clean_tile_at_position(self, pos, cleaning_volume):
        """
        Mark the tile under the position pos as cleaned by cleaning_volume amount of dust.

        Assumes that pos represents a valid position inside this room.

        pos: a Position object
        cleaning_volume: a float, the amount of dust to be cleaned in a single time-step.
                  Can be negative which would mean adding dust to the tile.

        Note: The amount of dust on each tile should be NON-NEGATIVE.
              If the cleaning_volume exceeds the amount of dust on the tile, mark it as 0.
        """
       # using math.floor(x) to always round down when converting to ensure that Positions are always in the room.
       # Mark the tile under the position pos as 0 by cleaning_volume amount of dust if the volume is bigger than the amount of dust.
        if cleaning_volume >= self.tile[(math.floor(pos.x), math.floor(pos.y))]:
            self.tile[(math.floor(pos.x), math.floor(pos.y))] = 0
            # clean as much as possible and keep track of the remaining dust
        else:
            self.tile[(math.floor(pos.x), math.floor(pos.y))] -= cleaning_volume
        

    def is_tile_cleaned(self, w, h):
        """
        Return True if the tile (w, h) has been cleaned.

        Assumes that (w, h) represents a valid tile inside the room.

        w: an integer
        h: an integer

        Returns: True if the tile (w, h) is cleaned, False otherwise

        Note: The tile is considered clean only when the amount of dust on this
              tile is 0.
        """
        # clean : dust = 0
        if self.get_dust_amount(w, h) == 0:
            return True
        else: 
            return False

    def get_num_cleaned_tiles(self):
        """
        Returns: an integer; the total number of clean tiles in the room
        """
        num = 0
        # iterate over the tiles to see which ones are clean
        for i in self.tile.keys():
            if self.tile[i] == 0:
                num += 1
        return num
                
        
    def is_position_in_room(self, pos):
        """
        Determines if pos is inside the room.

        pos: a Position object.
        Returns: True if pos is in the room, False otherwise.
        """
        if (math.floor(pos.x), math.floor(pos.y)) in self.tile.keys():
            return True
        else:
            return False

    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        return (self.width*self.height)

    def get_random_position(self):
        """
        Returns: a Position object; a random position inside the room
        """
        x = random.uniform(0,self.width)
        y = random.uniform(0,self.height)
        random_pos = Position(x,y)
        # check if the position is in the room
        if self.is_position_in_room(random_pos):
            return random_pos


class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times, the robot has a particular position and direction in the room.
    The robot also has a fixed speed and a fixed cleaning_volume.

    Subclasses of Robot should provide movement strategies by implementing
    update_position_and_clean, which simulates a single time-step.
    """
    def __init__(self, room, speed, cleaning_volume):
        """
        Initializes a Robot with the given speed and given cleaning_volume in the
        specified room. The robot initially has a random direction and a random
        position in the room.

        room:  a Room object.
        speed: a positive float.
        cleaning_volume: a positive float; the amount of dust cleaned by the robot
                  in a single time-step.
        """
        self.room = room
        self.speed = speed
        self.cleaning_volume = cleaning_volume
        self.position = self.room.get_random_position()
        self.direction = random.uniform(0, 360)
        
    def get_position(self):
        """
        Returns: a Position object giving the robot's position in the room.
        """
        return self.position

    def get_direction(self):
        """
        Returns: a float d giving the direction of the robot as an angle in
        degrees, 0.0 <= d < 360.0.
        """
        
        return self.direction
  

    def set_position(self, position):
        """
        Set the position of the robot to position.

        position: a Position object.
        """
        self.position = position
        return self.position
        
        
    def set_direction(self, direction):
        """
        Set the direction of the robot to direction.

        direction: float representing an angle in degrees clockwise from north
        """
        self.direction = direction
        return self.direction

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Moves robot to new position and cleans tile according to robot movement
        rules.
        """
        # DO NOT CHANGE -- implement in subclasses
        raise NotImplementedError

# === Problem 2
class NormalRobot(Robot):
    """
    A NormalRobot is a Robot with the standard movement strategy.

    At each time-step, a NormalRobot attempts to move in its current
    direction; when it would hit a wall, it *instead*
    chooses a new direction randomly.
    """
    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Calculate the next position for the robot.

        If that position is valid, move the robot to that position. Mark the
        tile it is on as having been cleaned by cleaning_volume amount.

        If the new position is invalid, do not move or clean the tile, but
        rotate once to a random new direction.
        """

        new_position = self.get_position().get_new_position(self.get_direction(), self.speed)
# # If that is a valid position(in the room), move there and then clean the tile corresponding to that position by the robot???s cleaning volume.
        if self.room.is_position_in_room(new_position):
            self.set_position(new_position)
            self.room.clean_tile_at_position(self.position, self.cleaning_volume)
# Otherwise, rotate the robot to be pointing in a random new direction. Don???t clean the current tile or move to a different tile.
        else:
            self.set_direction(random.uniform(0, 360))
            
            

        
        
# === Problem 3
class ClumsyRobot(Robot):
    """
    A ClumsyRobot is a robot that may accidentally drop dust on a tile. A ClumsyRobot will drop some dust 
    on the tile it's on with probability p. 
    The amount of dropped dust should be a random decimal value between 0 and 0.5. 
    Regardless of whether the robot drops dust, it moves to a new position to clean that tile.
    If that new position is not valid, the robot just randomly changes direction.
    """
    p = 0.05

    @staticmethod
    def set_dust_probability(prob):
        """
        Sets the probability of the robot accidentally dropping dust on the tile equal to prob.

        prob: a float (0 <= prob <= 1)
        """
        ClumsyRobot.p = prob

    def does_drop_dust(self):
        """
        Answers the question: Does the robot accidentally drop dust on the tile
        at this timestep?
        The robot drops dust with probability p.

        returns: True if the robot drops dust on its tile, False otherwise.
        """
        return random.random() < ClumsyRobot.p

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        1. Before moving, the robot checks if it should drop dust using does_drop_dust.
            If the robot drops dust, it adds a random decimal amount of dust
            between 0 (inclusive) and 0.5 (exclusive).
            If the robot does not drop dust, it just goes to step 2.
        2. Calculate the robot's next position.
            If the position is valid, the robot cleans that tile.
            If the position is not valid, the robot randomly picks a new direction.
        """
 
# check if the robot does drop dust
# the amount of dust that will be dropped is a random decimal value between 0 (inclusive) and 0.5 (exclusive). 

        if self.does_drop_dust(): 
            self.room.clean_tile_at_position(self.position, -random.uniform(0,0.5))
        new_position = self.get_position().get_new_position(self.get_direction(), self.speed)
# If that is a valid position(in the room), move there and then clean the tile corresponding to that position by the robot???s cleaning volume.
        if self.room.is_position_in_room(new_position):
            self.set_position(new_position)
            self.room.clean_tile_at_position(self.position, self.cleaning_volume)
# Otherwise, rotate the robot to be pointing in a random new direction. Don???t clean the current tile or move to a different tile.
               
        else:
            self.set_direction(random.uniform(0, 360))             


# === Problem 4
class SensingRobot(Robot):
    """
    A SensingRobot is a robot that can decide which direction to go based on the amount of dust it 
    sees.

    In one timestep, the SensingRobot will look at its surrounding area and find where there is the
    most dust. In the same timestep, it will move towards one of the positions with the most dust.

    It scans the surrounding area by checking each angles between 0 and 360.
    Naturally, many scans will return the same amount of dust, so after the robot completes it's 
    scan, it will randomly pick with uniform probability one of the angles that have the most 
    amount of dust.
    """

    def sense_dust_at_angle(self, angle):
        """
        args:
            angle (int): Angle in degrees of which direction to look

        returns:
            The amount of dust at the position the robot would end up if it moved in the direction
            of angle. Returns -1 if the position is a wall.

        DO NOT MODIFY
        """
        lookahead_pos = self.get_position().get_new_position(angle, self.speed)
        if not self.room.is_position_in_room(lookahead_pos):
            return -1

        tile = (int(lookahead_pos.get_x()), int(lookahead_pos.get_y()))
        return self.room.get_dust_amount(tile[0], tile[1])

    def scan_surrounding_area(self):
        """
        Looks at every 5th integer angle from [0, 360) and returns a dictionary mapping angle to
        dust amount.

        If the scanner sees a wall in one of the scans, it will give a reading of -1.

        DO NOT MODIFY
        """
        dust_amounts = {}
        for angle in range(0, 360, 5):
            dust_amounts[angle] = self.sense_dust_at_angle(angle)

        return dust_amounts

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Within one time step (i.e. one call to update_position_and_clean), the robot should:

        1. Scan the surrounding area (use scan_surrounding_area)
        2. Find the angles with the maximum amount of dust
        3. Pick one of the dirtiest angles at random and move in that direction (random.choice()
           might be useful!)
        4. Clean the tile the robot lands on
        """
        # Scan the surrounding area (use scan_surrounding_area)
        dusts = self.scan_surrounding_area()
        # Find the angles with the maximum amount of dust
        max_dust = max(dusts.values())
        # Pick one of the dirtiest angles at random
        dirty_angles = []
        for i in dusts.keys():
            if dusts[i] == max_dust:
                dirty_angles.append(i)
        pick = random.choice(dirty_angles)
        # move in that direction
        self.direction = pick
        # get the new position
        new_position = self.get_position().get_new_position(self.direction, self.speed)
# # If that is a valid position(in the room), move there and then clean the tile corresponding to that position by the robot???s cleaning volume.
        if self.room.is_position_in_room(new_position):
            self.set_position(new_position)
            self.room.clean_tile_at_position(self.position, self.cleaning_volume)
# Otherwise, rotate the robot to be pointing in a random new direction. Don???t clean the current tile or move to a different tile.
        else:
            self.set_direction(self.direction)
        

# === Problem 5
def run_simulation(num_robots, speed, cleaning_volume, width, height, dust_amount, min_coverage, num_trials,
                  robot_type):
    """
    Runs num_trials trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction min_coverage of the room.

    The simulation is run with num_robots robots of type robot_type, each
    with the input speed and cleaning_volume in a room of dimensions width x height
    with the dust dust_amount on each tile. Each trial is run in its own Room
    with its own robots.

    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    cleaning_volume: a float (cleaning_volume > 0)
    width: an int (width > 0)
    height: an int (height > 0)
    dust_amount: an int
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. NormalRobot or
                ClumsyRobot)
    """
    trials = []
    for trial in range(num_trials):
        # initialize a new room every time 
        room = Room(width, height, dust_amount)
        # create the robots
        robots = []
        for r in range(num_robots): 
            # use robot_type so that each type of robots can be called easily
            robot = robot_type(room, speed, cleaning_volume)
            robots.append(robot)
        # set a counter for time_steps
        time_steps = 0
        cleaned_fraction = room.get_num_cleaned_tiles()/room.get_num_tiles()
        # run the robots till the min_coverage is fulfilled
        while cleaned_fraction < min_coverage:
            for robot in robots:
                robot.update_position_and_clean()
            time_steps += 1
            cleaned_fraction = room.get_num_cleaned_tiles()/room.get_num_tiles()
        trials.append(time_steps)
# The average number of time steps for the simulation = the sum of the time steps across all the trials/ the number of trials.
    return sum(trials)/len(trials)
        
            



# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 5, 5, 3, 1.0, 50, NormalRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.8, 50, NormalRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.9, 50, NormalRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 20, 20, 3, 0.5, 50, NormalRobot)))
# print ('avg time steps: ' + str(run_simulation(3, 1.0, 1, 20, 20, 3, 0.5, 50, NormalRobot)))

# === Problem 6
#
# ANSWER THE FOLLOWING QUESTIONS:
#
# 1)How does the performance of the three robot types compare when cleaning 80%
#       of a 20x20 room?
#
#  Firstly, the more robots there are, the faster the robots finish the cleaning task80% of a 20x20 room.
#  Among the three types, sensing robot clean the room most (much more)efficiently than the other two, followed by normal robots
# clumsy robots did the worst job.  
    
# 2) How does the performance of the three robot types compare when two of each
#       robot cleans 80% of rooms with dimensions
#       10x30, 20x15, 25x12, and 50x6?

# the sensing robots clean the room in the shortest time steps, followed by normal robots, and then clumsy robots. 

# The two sensing robots have the best performance and they cleaned the room with 20/15 aspect ratio with the fastest speed.

# When the ratio aspect is smaller than 20/15,the speed at which robots clean the room increases; 
# 
# When aspect ratio is bigger than 20/15,the speed at which robots clean the room decreases; 
#
# Aspect ratio has the biggest impact on sensing robots but relatively small impacts on clumsy robots. 
# Aspect ratio has almost no effect on normal robots.

#

def show_plot_compare_strategies(title, x_label, y_label):
    """
    Produces a plot comparing the three robot strategies in a 20x20 room with 80%
    minimum coverage.
    """
    num_robot_range = range(1, 3)
    times1 = []
    times2 = []
    times3 = []
    for num_robots in num_robot_range:
        print ("Plotting", num_robots, "robots...")
        times1.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, NormalRobot))
        times2.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, ClumsyRobot))
        times3.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, SensingRobot))
    pylab.plot(num_robot_range, times1)
    pylab.plot(num_robot_range, times2)
    pylab.plot(num_robot_range, times3)
    pylab.title(title)
    pylab.legend(('NormalRobot', 'ClumsyRobot', 'SensingRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()

def show_plot_room_shape(title, x_label, y_label):
    """
    Produces a plot showing dependence of cleaning time on room shape.
    """
    aspect_ratios = []
    times1 = []
    times2 = []
    times3 = []
    for width in [10, 20, 25, 50]:
        height = int(300/width)
        print ("Plotting cleaning time for a room of width:", width, "by height:", height)
        aspect_ratios.append(float(width) / height)
        times1.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, NormalRobot))
        times2.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, ClumsyRobot))
        times3.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, SensingRobot))
    pylab.plot(aspect_ratios, times1, 'o-')
    pylab.plot(aspect_ratios, times2, 'o-')
    pylab.plot(aspect_ratios, times3, 'o-')
    pylab.title(title)
    pylab.legend(('NormalRobot', 'ClumsyRobot', 'SensingRobot'), fancybox=True, framealpha=0.5)
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()

if __name__ == "__main__":
    # Test code should go HERE so that test_ps3.py doesn't have to run it!
    # All code in this block will be run ONLY if you run ps3.py directly.

    # Uncomment this line to see your implementation of NormalRobot in action!
    # test_robot_movement(NormalRobot, Room)

    # Uncomment this line to see your implementation of ClumsyRobot in action!
    # test_robot_movement(ClumsyRobot, Room)
    
    # Uncomment this line to see your implementation of SensingRobot in action!
    # test_robot_movement(SensingRobot, Room)

    # show_plot_compare_strategies('Time to clean 80% of a 20x20 room, for various numbers of robots','Number of robots','Time (steps)')
    # show_plot_room_shape('Time to clean 80% of a 300-tile room for various room shapes','Aspect Ratio', 'Time (steps)')

    pass
