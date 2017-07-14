import rng

class Particle:
    __position_i = []  # particle position
    __velocity_i = []  # particle velocity
    __pos_best_i = []  # best position individual
    __err_best_i = -1  # best error individual
    __err_i = -1  # error individual
    __num_dimensions = 2
    __ng = rng.RNG("Random")

    def __init__(self,x0, num_dimensions):
        self.__num_dimensions = num_dimensions

        for i in range(0, self.__num_dimensions):
            self.__velocity_i.append(self.__ng.uniform(-1,1))
            self.__position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.__err_i=costFunc(self.__position_i)

        # check to see if the current position is an individual best
        if self.__err_i < self.__err_best_i or self.__err_best_i==-1:
            self.__pos_best_i=self.__position_i
            self.__err_best_i=self.__err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in ra__nge(0,self.__num_dimensions):
            r1=self.__ng.random()
            r2=self.__ng.random()

            vel_cognitive=c1*r1*(self.__pos_best_i[i]-self.__position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.__position_i[i])
            self.__velocity_i[i]=w*self.__velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,self.__num_dimensions):
            self.__position_i[i]=self.__position_i[i]+self.__velocity_i[i]

            # adjust maximum position if necessary
            if self.__position_i[i]>bounds[i][1]:
                self.__position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.__position_i[i] < bounds[i][0]:
                self.__position_i[i]=bounds[i][0]