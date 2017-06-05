"""
http://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html

Step 2: Calculate the error gradient w.r.t the weights

∂SSE/∂a = – (Y-YP)

∂SSE/∂b = – (Y-YP)X

Here, SSE=½ (Y-YP)2 = ½(Y-(a+bX))2

You need to know a bit of calculus, but that’s about it!!

∂SSE/∂a and ∂SSE/∂b are the gradients and they give the direction of the movement of a,b w.r.t to SSE.


Step 3:Adjust the weights with the gradients to reach the optimal values where SSE is minimized

We need to update the random values of a,b so that we move in the direction of optimal a, b.

Update rules:

a – ∂SSE/∂a
b – ∂SSE/∂b
So, update rules:
3.300 = ∂SSE/∂a = – (Y-YP) = sum of – (Y-YP) for each data point
1.545 = ∂SSE/∂b = – (Y-YP)X = sum of – (Y-YP)X for each data point
New a = a – r * ∂SSE/∂a = 0.45-0.01*3.300 = 0.42
New b = b – r * ∂SSE/∂b= 0.75-0.01*1.545 = 0.73
here, r is the learning rate = 0.01, which is the pace of adjustment to the weights.

Step 4:Use new a and b for prediction and to calculate new Total SSE


"""


from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run()
