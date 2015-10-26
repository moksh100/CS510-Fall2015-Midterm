import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Attractor(object):
        
        def __init__(self,  s=10, p=28, b=2.66667, start=0.0, end=80.0, points=10000):
                self.params = np.array([s,p,b])
                self.start = start 
                self.end = end
                self.points = points
                self.dt = (end-start)/points

        def euler (self, ar = np.array([]), dt=0):
                if(dt==0):
                    dt =self.dt
                x=ar[0]
                y=ar[1]
                z=ar[2]
                dz = (x*y)-(self.params[2]*z)
                dy = x*(self.params[1]-z)-y
                dx = self.params[0]*(y-x)
                xdt=x+(dx*dt)
                ydt=y+(dy*dt)
                zdt=z+(dz*dt)
                return np.array([xdt,ydt,zdt])

        def rk2 (self, ar = np.array([])):
                dt=self.dt/2
                k1= self.euler(ar)
                x=ar[0]+k1[0]*dt
                y=ar[1]+k1[1]*dt
                z=ar[2]+k1[2]*dt
                calc = self.euler(np.array([x,y,z]), dt)
                return calc

        def rk3 (self, ar = np.array([])):
                dt=self.dt/2
                k2= self.rk2(ar)
                x=ar[0]+k2[0]*dt
                y=ar[1]+k2[1]*dt
                z=ar[2]+k2[2]*dt
                calc = self.euler(np.array([x,y,z]), dt)
                return calc

        def rk4 (self, ar = np.array([])):
                dt=self.dt
                k3= self.rk3(ar)
                x=ar[0]+k3[0]*dt
                y=ar[1]+k3[1]*dt
                z=ar[2]+k3[2]*dt
                calc = self.euler(np.array([x,y,z]), dt)
                return calc
            
            
        def evolve(self, r0= np.array([0.1, 0.0, 0.0]), order =4):
            iterator=np.arange(self.start+1, self.end, self.dt)
            iterator=np.append(iterator, self.end)
            result = np.array(np.append(0, r0))
            v=r0;
            if(order == 4):
                for t in iterator:
                    v = self.rk4(v)
                    result = np.vstack((result,np.append(t, v)))
            elif(order == 3):
                for t in iterator:
                    v = self.rk3(v)
                    result = np.vstack((result,np.append(t, v)))
            elif(order == 2):
                for t in iterator:
                    v = self.rk2(v)
                    result = np.vstack((result,np.append(t, v)))
            elif(order == 1):
                for t in iterator:
                    v = self.euler(v)
                    result = np.vstack((result,np.append(t, v)))
            
            df = pd.DataFrame(result)
            df.columns = ['t', 'x', 'y', 'z']
            self.solution = df
            return df        
        
        
        def save(self):
            self.solution.to_csv("export.csv")
            
        def plotx(self):
            self.solution['x'].plot()
            plt.show()
            
        def ploty(self):
            self.solution['y'].plot()
            plt.show()
                        
        def plotz(self):
            self.solution['z'].plot()
            plt.show()
            
        def plotxy(self):
            plt.plot(self.solution['t'],self.solution['x'],'g')
            plt.plot(self.solution['t'],self.solution['y'],'r')
            plt.show()
        
        def plotyz(self):
            plt.plot(self.solution['t'],self.solution['y'],'g')
            plt.plot(self.solution['t'],self.solution['z'],'r')
            plt.show()
        
        def plotzx(self):
            plt.plot(self.solution['t'],self.solution['z'],'g')
            plt.plot(self.solution['t'],self.solution['x'],'r')
            plt.show()
           
        def plot3d(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.solution['x'], self.solution['y'], self.solution['z'])
            plt.show()