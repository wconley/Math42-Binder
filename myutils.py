from functools import partial
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly
from scipy.integrate import RK23
from ipywidgets import interact as _original_interact, HBox, Layout

def interact(_function_to_wrap=None, _layout="horizontal", **kwargs):
    """interact, but with widgets laid out in a horizontal flexbox layout

    This function works exactly like 'interact' (from ipywidgets or SageMath), 
    except that instead of putting all of the widgets into a vertical box 
    (VBox), it uses a horizontal box (HBox) by default. The HBox uses a flexbox 
    layout, so that if there are many widgets, they'll wrap onto a second row. 

    Options:
        '_layout' - 'horizontal' by default. Anything else, and it will revert 
        back to using the default layout of 'interact' (a VBox). 
    """
    def decorator(f):
        retval = _original_interact(f, **kwargs)
        if _layout == "horizontal":
            widgets = retval.widget.children[:-1]
            output = retval.widget.children[-1]
            hbox = HBox(widgets, layout=Layout(flex_flow="row wrap"))
            retval.widget.children = (hbox, output)
        return retval
    if _function_to_wrap is None:
        # No function passed in, so this function must *return* a decorator
        return decorator
    # This function was called directly, *or* was used as a decorator directly
    return decorator(_function_to_wrap)


def latex_matrix(m, round=None):
    if round is not None:
        m = np.round(m, round)
    m = np.array(m, dtype=str)
    rows = [" & ".join(row) for row in m]
    matrix = r" \\ ".join(rows)
    return r"\begin{pmatrix} " + matrix + r" \end{pmatrix}"


def latex_vector(v, round=None):
    if round is not None:
        v = np.round(v, round)
    v = np.array(v, dtype=str)
    vector = r" \\ ".join(v)
    return r"\begin{bmatrix} " + vector + r" \end{bmatrix}"


def latex(x, round=None):
    try:
        x = np.array(x)
        shape = x.shape
    except:
        shape = None
    if shape is None or len(shape) < 1 or len(shape) > 2:
        raise ValueError("argument is not a vector nor a matrix")
    if len(shape) == 1:
        return latex_vector(x, round)
    if len(shape) == 2:
        return latex_matrix(x, round)


def ddeint(f, history, tmax, tmin=0, **options):
    """numerically integrate a delay differential equation via Bogacki–Shampine

    This uses the Bogacki–Shampine formulas, an explicit Runge–Kutta method of 
    order 3(2) with variable step size. The solution is returned as a cubic 
    Hermite spline. 

    Arguments: 
        'f' - the function that defines the DDE. The signature of this function 
        is 
            yprime = f(t, y)
        where yprime may be either a scalar, or a vector quantity (tuple, list, 
        numpy array, etc) of some length n >= 1, in which case the argument y 
        will be the same. y (or each component of y if it's a vector) is a 
        function, so that 'f' can access the history of each state variable. 
        Example: 
            def f(t, y):
                N, P = y
                return (0.5*N(t) - 0.2*N(t)*P(t), 0.1*N(t-2)*P(t-2) - 0.4*P(t))
        'history' - the history function that defines the initial state. The 
        signature of this should be 
            y = history(t)
        where y must have the same type (scalar, or vector of length n) as the 
        output yprime and input y of 'f'. 
        'tmax' - the t value to run the simulation up to. 
    Options: 
        'tmin' - the starting t value. Defaults to 0. 
        All other options are passed to the underlying scipy.integrate.RK23 as 
        is (defaults are 'max_step': inf, 'rtol': 0.001, 'atol': 1e-06, 
        'first_step': None), except that the 'vectorized' option is always set 
        to False. We note in particular the 'max_step' argument. It is 
        *strongly* recommended that you set 'max_step' equal to something less 
        than the smallest delay of your DDE. Otherwise, the solver is likely to 
        raise an exception. 
    Returns:
        A cubic Hermite spline (as a scipy.interpolate.PPoly instance) 
        representing the solution, from 'tmin' up to (and possibly slightly 
        beyond) 'tmax'. Note that this can be called like a function (which is 
        useful for plotting using an adaptive function plotter), and it's a 
        vectorized function, so it's very easy to sample a bunch of points from 
        it, for example to plot the results as a scatter plot. 
    """
    options["vectorized"] = False
    interpolator = None
    initial_y = history(tmin)
    try:
        n = len(initial_y)
        f_scalar = False
    except:
        n = 1
        initial_y = (initial_y,)
        f_scalar = f
        f = lambda t, y: (f_scalar(t, y[0]),)
        history_scalar = history
        history = lambda t: (history_scalar(t),)
    def f_(current_t, current_y):
        def y(i, t):
            if t == current_t:
                return current_y[i]
            if t <= tmin:
                return history(t)[i]
            if interpolator is None or interpolator.x[-1] < t:
                raise ValueError("Time step larger than delay!")
            return interpolator(t)[i]
        return f(current_t, [partial(y, i) for i in range(n)])
    solver = RK23(f_, tmin, initial_y, tmax, **options)
    solver.step()
    ts = (tmin, solver.t)
    ys = (initial_y, solver.y)
    yprimes = (f_(tmin, initial_y), f_(solver.t, solver.y))
    interpolator = CubicHermiteSpline(ts, ys, yprimes)
    while solver.status == "running":
        solver.step()
        ts = (ts[1], solver.t)
        ys = (ys[1], solver.y)
        yprimes = (yprimes[1], f_(solver.t, solver.y))
        new_interpolator = CubicHermiteSpline(ts, ys, yprimes)
        interpolator.extend(new_interpolator.c, new_interpolator.x[1:])
    if f_scalar:
        shape = interpolator.c.shape[:-1]
        return PPoly(interpolator.c.reshape(shape), interpolator.x)
    return interpolator

