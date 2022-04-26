import warnings
from random import randrange

import numpy as np
import plotly.graph_objects
rng = np.random.default_rng()

ARROW_STYLES = {
    # The last two parameters below are for an arrow pointing straight up. 
    # tipfactor: Arrowhead length, as a fraction of the total arrow length
    # tipslope1: slope of the leading edge of the left side of the arrowhead
    # tipslope2: slope of the trailing edge of the left side of the arrowhead
    "stealth":      { "tipfactor": 0.1 , "tipslope1": 1.0, "tipslope2": 0.5 }, 
    "stealth-big":  { "tipfactor": 0.2 , "tipslope1": 1.0, "tipslope2": 0.5 }, 
    "pointy":       { "tipfactor": 0.1 , "tipslope1": 2.5, "tipslope2": 1.0 }, 
    "pointy-big":   { "tipfactor": 0.2 , "tipslope1": 2.5, "tipslope2": 1.0 }, 
    "latex":        { "tipfactor": 0.1 , "tipslope1": 2.0, "tipslope2": 0.0 }, 
    "latex-big":    { "tipfactor": 0.2 , "tipslope1": 2.0, "tipslope2": 0.0 }, 
    "straightbarb": { "tipfactor": 0.2 , "tipslope1": 1.0, "tipslope2": 1.0 }, 
}
default_arrow_style = None

def set_defaults(**options):
    global default_arrow_style
    default_arrow_style = options.get("arrow_style", "pointy-big")

    mydefault = plotly.graph_objects.layout.Template()
    # Set default size
    mydefault.layout.width  = options.get("width")  # Unsets it if it's None
    mydefault.layout.height = options.get("height") # Unsets it if it's None
    # Set default margins, CSS-style: all, (tb, rl), (t, rl, b), or (t, r, b, l)
    margins = options.get("margins", options.get("margin"))
    if margins is not None:
        try:
            margins = (int(margins), ) * 4
        except TypeError:
            pass
        if len(margins) == 2:
            margins = tuple(margins) * 2
        elif len(margins) == 3:
            margins = tuple(margins) + (margins[1], )
        mydefault.layout.margin = dict(zip("trbl", margins))

    # Whether or not to show the grid for 2D plots (defaults to both off)
    grid = options.get("grid", False)
    if grid == "both" or grid == True:
        grid = (True, True)
    elif grid == "x":
        grid = (True, False)
    elif grid == "y":
        grid = (False, True)
    else:
        grid = (False, False)
    mydefault.layout.xaxis.showgrid, mydefault.layout.yaxis.showgrid = grid

    # Whether or not to show the axis lines for 2D plots (defaults to both on)
    axislines = options.get("axislines", True)
    if axislines == "both" or axislines == True:
        axislines = (True, True)
    elif axislines == "x":
        axislines = (True, False)
    elif axislines == "y":
        axislines = (False, True)
    else:
        axislines = (False, False)
    mydefault.layout.xaxis.showline, mydefault.layout.yaxis.showline = axislines

    # Possibilities for future customization: ticks, zeroline, zerolinewidth
    #mydefault.layout.xaxis.ticks = "outside"
    #mydefault.layout.yaxis.ticks = "outside"
    #mydefault.layout.xaxis.zeroline = True
    #mydefault.layout.yaxis.zeroline = True
    #mydefault.layout.xaxis.zerolinewidth = 2
    #mydefault.layout.yaxis.zerolinewidth = 2

    # Set default drag mode. Plotly default is zoom. My default is pan. 
    mydefault.layout.dragmode = options.get("dragmode", "pan")

    # Turn off spikes (hover lines that go from plot points to axes)
    mydefault.layout.hovermode = False
    mydefault.layout.scene.hovermode = False
    mydefault.layout.xaxis.showspikes = False
    mydefault.layout.yaxis.showspikes = False
    mydefault.layout.scene.xaxis.showspikes = False
    mydefault.layout.scene.yaxis.showspikes = False
    mydefault.layout.scene.zaxis.showspikes = False

    # Turn off certain default features for specific types of plots
    mydefault.data.contour = [plotly.graph_objects.Contour(
            showscale=False, 
    )]
    mydefault.data.scatter3d = [plotly.graph_objects.Scatter3d(
            hoverinfo="skip", 
    )]
    mydefault.data.surface = [plotly.graph_objects.Surface(
            showscale=False, 
            hoverinfo="skip", 
            contours_x_highlight=False, 
            contours_y_highlight=False, 
            contours_z_highlight=False, 
    )]
    mydefault.data.isosurface = [plotly.graph_objects.Isosurface(
            showscale=False, 
            hoverinfo="skip", 
    )]
    mydefault.data.cone = [plotly.graph_objects.Cone(
            showscale=False, 
            hoverinfo="skip", 
    )]

    # Install our template as the default
    plotly.io.templates["mydefault"] = mydefault
    plotly.io.templates.default = "mydefault"


# Actually call the above function, to install those defaults!
set_defaults()


# Set up a config that can be passed to Figure.show(config=myconfig)
# Unfortunately, this doesn't work with a FigureWidget at all, because show() 
# doesn't seem to work with a FigureWidget, and there doesn't seem to be any 
# other way to pass this information to a FigureWidget. But this might still be 
# useful for non-widget Figures. 
#plot_default_config={"scrollZoom":     True, 
#                     "displayModeBar": True, 
#                     "displaylogo":    False, 
#                     "showTips":       False, 
#                     "modeBarButtonsToRemove": ["select2d", 
#                                                "lasso2d", 
#                                                "toggleSpikelines", 
#                                                "hoverClosestCartesian", 
#                                                "hoverCompareCartesian", 
#                                               ]}


# Our proxy object for a single “subplot” (axes) within a Figure or FigureWidget
class PlotlyAxes(object):
    ANNOTATION, ANNOTATION3D, OTHER = range(3)

    def __init__(self, figure, row, col):
        self._figure = figure
        self._row = row
        self._col = col
        self._items = {}
        self._aspect_ratio = None

    def _add_item(self, item):
        if isinstance(item, (tuple, list)):
            index = [self._add_item(i) for i in item]
        elif isinstance(item, plotly.graph_objects.layout.Annotation):
            index = 3*len(self._figure.layout.annotations) + self.ANNOTATION
            if item.xref == "paper" and item.yref == "paper":
                self._figure.add_annotation(item)
            else:
                self._figure.add_annotation(item, row=self._row, col=self._col)
        elif isinstance(item, plotly.graph_objects.layout.scene.Annotation):
            subplot = self._figure.get_subplot(self._row, self._col)
            index = 3*len(subplot.annotations) + self.ANNOTATION3D
            subplot.annotations += (item,)
        else:
            index = 3*len(self._figure.data) + self.OTHER
            self._figure.add_trace(item, row=self._row, col=self._col)
        return index

    def _update_item(self, index, item):
        if isinstance(index, (tuple, list)):
            for i in index:
                self._update_item(i, item)
            return
        index, kind = divmod(index, 3)
        if kind == self.ANNOTATION:
            self._figure.layout.annotations[index].update(item)
        elif kind == self.ANNOTATION3D:
            subplot = self._figure.get_subplot(self._row, self._col)
            subplot.annotations[index].update(item)
        elif kind == self.OTHER:
            self._figure.data[index].update(item)

    def __setattr__(self, name, value):
        if name[0] == "_":
            object.__setattr__(self, name, value)
        elif name in self.__dict__:
            raise AttributeError(f"Cannot overwrite existing attribute {name}")
        else:
            index = self._items.get(name)
            if index is None:
                index = self._add_item(value)
                self._items[name] = index
            else:
                self._update_item(index, value)

    def __getattr__(self, name):
        index = self._items.get(name)
        if index is None:
            raise AttributeError(f"this axes object has no attribute '{name}'")
        index, kind = divmod(index, 3)
        if kind == self.ANNOTATION:
            return self._figure.layout.annotations[index]
        if kind == self.ANNOTATION3D:
            subplot = self._figure.get_subplot(self._row, self._col)
            return subplot.annotations[index]
        if kind == self.OTHER:
            return self._figure.data[index]

    def axes_labels(self, *labels):
        subplot = self._figure.get_subplot(self._row, self._col)
        numaxes = 3 if hasattr(subplot, "zaxis") else 2
        if len(labels) != numaxes:
            raise ValueError(f"expected {numaxes} labels but got {len(labels)}")
        subplot.xaxis.title.text = labels[0]
        subplot.yaxis.title.text = labels[1]
        if numaxes == 3:
            subplot.zaxis.title.text = labels[2]

    def axes_ranges(self, *ranges, scale=None):
        subplot = self._figure.get_subplot(self._row, self._col)
        numaxes = 3 if hasattr(subplot, "zaxis") else 2
        if len(ranges) != numaxes:
            raise ValueError(f"expected {numaxes} ranges but got {len(ranges)}")
        if ranges[0] == "off": # First deal with the x-axis
            subplot.xaxis.showline = False
            subplot.xaxis.showticklabels = False
            subplot.yaxis.zeroline = False
            if len(ranges) == 3:
                subplot.zaxis.zeroline = False
        elif isinstance(ranges[0], str): # "normal", "tozero", or "nonnegative"
            subplot.xaxis.rangemode = ranges[0]
        else: # Otherwise, range should be a 2-tuple (or list) of numbers
            subplot.xaxis.range = ranges[0]
        if ranges[1] == "off": # First deal with the y-axis
            subplot.yaxis.showline = False
            subplot.yaxis.showticklabels = False
            subplot.xaxis.zeroline = False
            if len(ranges) == 3:
                subplot.zaxis.zeroline = False
        elif isinstance(ranges[1], str): # "normal", "tozero", or "nonnegative"
            subplot.yaxis.rangemode = ranges[1]
        else: # Otherwise, range should be a 2-tuple (or list) of numbers
            subplot.yaxis.range = ranges[1]
        if len(ranges) == 3: # Then deal with the z-axis if applicable
            if ranges[2] == "off":
                subplot.zaxis.showline = False
                subplot.zaxis.showticklabels = False
                subplot.xaxis.zeroline = False
                subplot.yaxis.zeroline = False
            elif isinstance(ranges[2], str):
                subplot.zaxis.rangemode = ranges[2]
            else:
                subplot.zaxis.range = ranges[2]
        if scale is not None: # Then deal with the aspect ratio/scale ratio
            if len(ranges) == 2: # 2D case. NOTE: scale must be a 2-tuple
                if not (isinstance(scale, (tuple, list)) and len(scale) == 2):
                    raise ValueError(f"'scale' must be a 2-tuple, got {scale}")
                x, y = scale
                subplot.xaxis.constrain = "domain"
                subplot.yaxis.constrain = "domain"
                subplot.yaxis.scaleanchor = "x"
                subplot.yaxis.scaleratio = y / x
                self._aspect_ratio = y / x
            elif isinstance(scale, str): # 3D case, one of the automatic modes
                subplot.aspectmode = scale
            elif isinstance(scale, (tuple, list)) and len(scale) == 3:
                # 3D case, manually specified aspect ratio. NOTE: In this case, 
                # all three axis ranges must be manually specified also. 
                (xmax, xmin), (ymax, ymin), (zmax, zmin) = ranges
                x, y, z = scale
                x *= xmax - xmin
                y *= ymax - ymin
                z *= zmax - zmin
                c = sorted((x, y, z))[1]
                subplot.aspectmode = "manual"
                subplot.aspectratio.update(x=x/c, y=y/c, z=z/c)
            else:
                raise ValueError(f"'scale' must be str or 3-tuple, got {scale}")

    @property
    def aspect_ratio(self):
        # If it was set in a call to axes_ranges(), return that
        if self._aspect_ratio is not None:
            return self._aspect_ratio
        subplot = self._figure.get_subplot(self._row, self._col)
        # NOT YET IMPLEMENTED: What to do for a 3D plot?
        if hasattr(subplot, "zaxis"):
            a = subplot.aspectratio
            return (a.x, a.y, a.z)
        # Otherwise, we attempt to compute it from known data, or guess
        x, y = self.resolution
        return y / x if x else None

    @property
    def resolution(self):
        subplot = self._figure.get_subplot(self._row, self._col)
        # NOT YET IMPLEMENTED: What to do for a 3D plot?
        if hasattr(subplot, "zaxis"):
            return (1, 1, 1)
        if subplot.xaxis.range is None or subplot.yaxis.range is None:
            return (None, None)
        def span(a): return a[1] - a[0]
        x, y = self._figure.layout.width, self._figure.layout.height
        if x is None or y is None:
            x, y = 400.0, 400.0 # These should be within an order of magnitude
        x *= span(subplot.xaxis.domain) / span(subplot.xaxis.range)
        y *= span(subplot.yaxis.domain) / span(subplot.yaxis.range)
        return (x, y)

# Our replacement for the make_subplots() function
def make_figure(rows=1, cols=1, **options):
    squeeze = options.pop("squeeze", True)
    widget = options.pop("widget", False)
    if widget:
        options.setdefault("figure", plotly.graph_objects.FigureWidget())
    figure = plotly.subplots.make_subplots(rows, cols, **options)
    subplots = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            if figure.get_subplot(i + 1, j + 1):
                subplots[i,j] = PlotlyAxes(figure, i + 1, j + 1)
    if squeeze:
        if rows == cols == 1:
            subplots = subplots[0,0]
        elif rows == 1 or cols == 1:
            subplots = subplots.flatten()
    return figure, subplots


# A function that optionally wraps a function f in a vectorizing wrapper
def vectorize(f, inputs, outputs):
    x = rng.random((inputs, 3))
    try:
        y = f(*x)
    except:
        pass
    else:
        if outputs == 1 and y.shape == (3,):
            return f
        if 1 < outputs == len(y) and all(y0.shape == (3,) for y0 in y):
            return f
    # If we get to this point, f does not work correctly on arrays, so vectorize
    signature = ",".join(["()"] * inputs) + "->"
    signature += "(n)" if outputs > 1 else "()"
    f_vec = np.vectorize(f, signature=signature)
    def f_(*args):
        return np.moveaxis(f_vec(*args), -1, 0)
    return f_


# Below are all the actual plotting methods. First, the 2D graphics:
def text(text, location, **options):
    options = options.copy()
    x, y = np.array(location, dtype=float)
    options.setdefault("font_color", options.pop("color", None))
    if (size := options.pop("size", None)) is not None:
        options.setdefault("font_size", size)
    if arrow := options.pop("arrow", False):
        options.setdefault("ax", float(arrow[0]))
        options.setdefault("ay", float(arrow[1]))
    options.setdefault("showarrow", bool(arrow))
    if options.pop("paper", False):
        options.update(xref="paper", yref="paper")
    if (anchor := options.pop("anchor", None)) is not None:
        if len(anchors := anchor.split()) > 1:
            yanchor, xanchor = anchors
        elif anchor in ("top", "middle", "bottom"):
            xanchor = "auto"
            yanchor = anchor
        elif anchor in ("left", "center", "right"):
            xanchor = anchor
            yanchor = "auto"
        else:
            raise ValueError(f"invalid value for anchor '{anchor}'")
        options.setdefault("xanchor", xanchor)
        options.setdefault("yanchor", yanchor)
    return plotly.graph_objects.layout.Annotation(
            text=text, x=x, y=y, **options)


def points(points, **options):
    options = options.copy()
    x, y = np.array(points, dtype=float).transpose()
    options.setdefault("marker_color", options.pop("color", None))
    options.setdefault("marker_size", options.pop("size", 8))
    options.setdefault("mode", "markers")
    return plotly.graph_objects.Scatter(x=x, y=y, **options)


def lines(points, **options):
    options = options.copy()
    x, y = np.array(points, dtype=float).transpose()
    options.setdefault("line_color", options.pop("color", None))
    options.setdefault("mode", "lines")
    if options.pop("smooth", False):
        options.setdefault("line_shape", "spline")
        options.setdefault("line_smoothing", 1.3)
    else:
        options.setdefault("line_shape", "linear")
        options.setdefault("line_smoothing", 0)
    return plotly.graph_objects.Scatter(x=x, y=y, **options)


def function(f, x_range, **options):
    options = options.copy()
    samples = options.pop("samples", 100)
    max_steps = options.pop("max_steps", 5)
    tolerance = options.pop("tolerance", 0.01)
    plot = options.pop("plot", None)
    aspect_ratio = plot.aspect_ratio if plot is not None else None
    aspect_ratio = options.pop("aspect_ratio", aspect_ratio)
    if aspect_ratio is None:
        aspect_ratio = 1
    options.setdefault("line_color", options.pop("color", None))
    options.setdefault("mode", "lines")
    if options.pop("smooth", False):
        options.setdefault("line_shape", "spline")
        options.setdefault("line_smoothing", 1.3)
    else:
        options.setdefault("line_shape", "linear")
        options.setdefault("line_smoothing", 0)
    # Compute the points for the plot, using an adaptive method
    f = vectorize(f, 1, 1)
    xmin, xmax = x_range
    x_gap = (xmax - xmin) / samples
    tolerance *= x_gap
    x = np.linspace(xmin, xmax, samples + 1, dtype=float)
    x[1:-1] += rng.uniform(-0.1*x_gap, 0.1*x_gap, samples - 1)
    points = np.array((x, f(x)))
    indices = np.arange(samples, dtype=int)
    for step in range(max_steps):
        midpoints = (points[:,indices] + points[:,indices + 1]) / 2
        midpoints_f = f(midpoints[0])
        dist = np.abs(midpoints_f - midpoints[1]) * aspect_ratio
        keep = (~np.isfinite(dist) | (dist > tolerance)).nonzero()[0]
        if keep.size == 0:
            break
        insertion = (midpoints[0,keep], midpoints_f[keep])
        keep = indices[keep]
        points = np.insert(points, keep + 1, insertion, axis=1)
        keep = keep + np.arange(keep.size)
        indices = np.empty(2*keep.size, dtype=int)
        indices[0::2] = keep
        indices[1::2] = keep + 1
    # Finally, produce the actual graph
    return plotly.graph_objects.Scatter(x=points[0], y=points[1], **options)


def parametric(f, t_range, **options):
    options = options.copy()
    samples = options.pop("samples", 100)
    max_steps = options.pop("max_steps", 5)
    tolerance = options.pop("tolerance", 0.5)
    plot = options.pop("plot", None)
    xres, yres = plot.resolution if plot is not None else (None, None)
    aspect_ratio = options.pop("aspect_ratio", 1)
    options.setdefault("line_color", options.pop("color", None))
    options.setdefault("mode", "lines")
    if options.pop("smooth", False):
        options.setdefault("line_shape", "spline")
        options.setdefault("line_smoothing", 1.3)
    else:
        options.setdefault("line_shape", "linear")
        options.setdefault("line_smoothing", 0)
    # Compute the points for the plot, using an adaptive method
    f = vectorize(f, 1, 2)
    tmin, tmax = t_range
    t_gap = (tmax - tmin) / samples
    if xres is None or yres is None:
        tolerance *= t_gap
        resolution = aspect_ratio ** np.array([[-0.5], [0.5]])
    else:
        resolution = np.array([[xres], [yres]], dtype=float)
    t = np.linspace(tmin, tmax, samples + 1, dtype=float)
    t[1:-1] += rng.uniform(-0.1*t_gap, 0.1*t_gap, samples - 1)
    points = np.array((t, (points := f(t))[0], points[1]))
    indices = np.arange(samples, dtype=int)
    for step in range(max_steps):
        midpoints = (points[:,indices] + points[:,indices + 1]) / 2
        midpoints_f = np.array(f(midpoints[0]))
        dist = np.linalg.norm((midpoints_f - midpoints[1:])*resolution, axis=0)
        keep = (~np.isfinite(dist) | (dist > tolerance)).nonzero()[0]
        if keep.size == 0:
            break
        insertion = (midpoints[0,keep], *midpoints_f[:,keep])
        keep = indices[keep]
        points = np.insert(points, keep + 1, insertion, axis=1)
        keep = keep + np.arange(keep.size)
        indices = np.empty(2*keep.size, dtype=int)
        indices[0::2] = keep
        indices[1::2] = keep + 1
    # Finally, produce the actual graph
    return plotly.graph_objects.Scatter(x=points[1], y=points[2], **options)


def contour(f, x_range, y_range, **options):
    options = options.copy()
    samples = options.pop("samples", 101)
    try:
        samplesx, samplesy = samples
    except:
        samplesx = samplesy = samples
    color = options.pop("color", None)
    if color is not None:
        options.setdefault("colorscale", (color, color))
        options.setdefault("contours_coloring", "lines")
    xmin, xmax = x_range
    ymin, ymax = y_range
    f = vectorize(f, 2, 1)
    x = np.linspace(xmin, xmax, samplesx)
    y = np.linspace(ymin, ymax, samplesy)
    z = f(*np.meshgrid(x, y))
    return plotly.graph_objects.Contour(x=x, y=y, z=z, **options)


def implicit(f, x_range, y_range, **options):
    options = options.copy()
    constant = options.pop("C", 0)
    options.setdefault("contours_start", constant)
    options.setdefault("contours_end", constant)
    options.setdefault("ncontours", 1)
    options.setdefault("line_width", 2)
    return contour(f, x_range, y_range, **options)


def vector(vec, start=(0, 0), axes_scale=(1, 1), **options):
    options = options.copy()
    arrow_style = options.pop(arrow_style, default_arrow_style)
    if isinstance(arrow_style, str):
        arrow_style = ARROW_STYLES[arrow_style]
    tipfactor = arrow_style["tipfactor"]
    tipslope1 = arrow_style["tipslope1"]
    tipslope2 = arrow_style["tipslope2"]
    color = options.pop("color", None)
    if color is not None:
        options.setdefault("line_color", color)
        options.setdefault("fillcolor", color)
    options.setdefault("line_width", 1)
    options["mode"] = "lines"
    options["fill"] = "toself"

    vec = np.array(vec, dtype=float)
    start = np.array(start, dtype=float)
    end = start + vec
    NaN = np.array((np.nan, np.nan), dtype=float)
    axes_scale = axes_scale[1] / axes_scale[0]
    tip = tipfactor * vec
    tip_perp = np.array(( -axes_scale/tipslope1 * tip[1], 
                         1/axes_scale/tipslope1 * tip[0]), dtype=float)
    tipleft = end - tip + tip_perp
    tipmiddle = end - float(1 - tipslope2/tipslope1) * tip
    tipright = end - tip - tip_perp
    xy = np.array((start, end, NaN, end, tipleft, tipmiddle, tipright, end))
    x, y = xy.transpose()
    return plotly.graph_objects.Scatter(x=x, y=y, **options)


def vector_field(f, x_range, y_range, **options):
    options = options.copy()
    axes_scale = options.pop("axes_scale", None)
    axes_aspect = options.pop("axes_aspect", (4, 3))
    scalefactor = options.pop("scalefactor", 1)
    region = options.pop("region", None)
    plotpoints = options.pop("plotpoints", 21)
    try:
        plotpointsx, plotpointsy = plotpoints
    except:
        plotpointsx = plotpointsy = plotpoints
    options["mode"] = "lines"
    options["fill"] = "toself"
    color = options.pop("color", "limegreen")
    options.setdefault("line_color", color)
    options.setdefault("fillcolor", color)
    options.setdefault("line_width", 1)

    # Arrow size/shape parameters
    zero_threshold = 1e-12
    minlength = 0.06 # Tip size won't be smaller than for an arrow of this length, as a fraction of the graph diagonal
    maxlength = 0.16 # Tip size won't be larger than for an arrow of this length, as a fraction of the graph diagonal
    tipfactor = 0.2  # Arrowhead length, as a fraction of the total arrow length
    tipslope1 = 2.5  # For an arrow pointing up, the slope of the leading edge of the left side of the arrowhead
    tipslope2 = 1.0  # For an arrow pointing up, the slope of the trailing edge of the left side of the arrowhead

    # Set the axes ranges, and the axes_scale and axes_aspect
    x, xmin, xmax = x_range
    y, ymin, ymax = y_range
    if axes_scale is None:
        axes_scale = axes_aspect[1] / axes_aspect[0] * (xmax - xmin) / (ymax - ymin)
    else:
        axes_scale = axes_scale[1] / axes_scale[0]
    axes_aspect = np.array((xmax - xmin, axes_scale * (ymax - ymin)), dtype=float)
    diagonal = np.linalg.norm(axes_aspect)

    # Choose the start and step size, for both x and y, for the grid points
    def grid_params(min1, max1, min2, max2, ratio):
        step1 = (max1 - min1) / (plotpoints + 0.75)
        start1 = min1 + 0.375 * step1
        step2 = step1 / ratio
        remainder = divmod(float((max2 - min2) / step2), 1)[1] / 2
        if remainder < 0.125:
            remainder += 0.5
        start2 = min2 + remainder * step2
        return start1, step1, start2, step2
    try:
        plotpoints[1]
    except: # A single value was given for plotpoints. Set grid automatically. 
        if axes_aspect[0] > axes_aspect[1]:
            x0, xstep, y0, ystep = grid_params(xmin, xmax, ymin, ymax, axes_scale)
        else:
            y0, ystep, x0, xstep = grid_params(ymin, ymax, xmin, xmax, 1/axes_scale)
    else: # A 2-tuple was given for plotpoints. Use exactly that many. 
        xstep = (xmax - xmin) / (plotpoints[0] + 0.75)
        x0 = xmin + 0.375 * xstep
        ystep = (ymax - ymin) / (plotpoints[1] + 0.75)
        y0 = ymin + 0.375 * ystep

    # The actual computations
    f1, f2 = [fast_float(f_, x, y) for f_ in f]
    xvals = np.arange(float(x0), float(xmax), float(xstep))
    yvals = np.arange(float(y0), float(ymax), float(ystep))
    xy = np.array(np.meshgrid(xvals, yvals)).reshape(2, -1).transpose()
    if region:
        region = fast_float(region, x, y)
        xy = xy[[region(x0, y0) > 0 for x0, y0 in xy]]
    vec = np.array([(f1(x0, y0), f2(x0, y0)) for x0, y0 in xy]).transpose()
    keep = np.linalg.norm(vec, axis=0) > zero_threshold
    xy = xy[keep].transpose()
    vec = vec[:,keep]
    grid_scale = np.array((xstep, ystep), dtype=float).reshape(2, 1)
    longest_vector = np.linalg.norm(vec / grid_scale, axis=0, ord=np.inf).max()
    vec *= scalefactor / longest_vector
    paper_scale = np.array((1, axes_scale), dtype=float).reshape(2, 1)
    length = np.linalg.norm(vec * paper_scale, axis=0) / diagonal
    tip = (tipfactor * np.clip(length, minlength, maxlength) / length) * vec
    tip_perp = np.array((float( -axes_scale/tipslope1) * tip[1], 
                         float(1/axes_scale/tipslope1) * tip[0]))
    NaN = np.full_like(xy, np.nan)
    end = xy + vec
    tipleft = end - tip + tip_perp
    tipmiddle = end - float(1 - tipslope2/tipslope1) * tip
    tipright = end - tip - tip_perp
    xy = np.array((xy, end, NaN, end, tipleft, tipmiddle, tipright, end, NaN))
    x, y = np.moveaxis(xy, 0, 2).reshape(2, -1)
    return plotly.graph_objects.Scatter(x=x, y=y, **options)


def trajectory(trajectory, **options):
    options = options.copy
    color = options.pop("color", "blue")
    arrows = options.pop("arrows", 6)
    options.setdefault("legendgroup", f"{randrange(1 << 32):08X}")
    options.setdefault("mode", "lines")
    options.setdefault("line_color", color)
    x, y = trajectory.transpose()
    p1 = plotly.graph_objects.Scatter(x=x, y=y, **options)

    del options["mode"]
    options["showlegend"] = False
    options.setdefault("colorscale", (color, color))
    options.setdefault("sizeref", 0.33) # Adjust this for smaller/largers cones
    try:
        arrows = np.array(list(arrows), dtype=int)
    except:
        arclength = np.cumsum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1))
        arrows_so_far = np.rint(arrows*arclength / arclength[-1])
        arrows = np.where(arrows_so_far[1:] - arrows_so_far[:-1])[0]
    arrow_pos = trajectory[arrows]
    arrow_dir = trajectory[arrows+1] - arrow_pos
    arrow_dir /= np.linalg.norm(arrow_dir, axis=1)[:,np.newaxis]
    x, y = arrow_pos.transpose()
    u, v = arrow_dir.transpose()
    #p2 = plotly.graph_objects.Cone(x=x, y=y, z=z, u=u, v=v, w=w, **options)
    return (p1, p2)


# Now all the plotting methods that produce 3D graphics:
def text3d(text, location, **options):
    options = options.copy()
    x, y, z = np.array(location, dtype=float)
    options.setdefault("font_color", options.pop("color", None))
    if (size := options.pop("size", None)) is not None:
        options.setdefault("font_size", size)
    if arrow := options.pop("arrow", False):
        options.setdefault("ax", float(arrow[0]))
        options.setdefault("ay", float(arrow[1]))
    options.setdefault("showarrow", bool(arrow))
    if (anchor := options.pop("anchor", None)) is not None:
        if len(anchors := anchor.split()) > 1:
            yanchor, xanchor = anchors
        elif anchor in ("top", "middle", "bottom"):
            xanchor = "auto"
            yanchor = anchor
        elif anchor in ("left", "center", "right"):
            xanchor = anchor
            yanchor = "auto"
        else:
            raise ValueError(f"invalid value for anchor '{anchor}'")
        options.setdefault("xanchor", xanchor)
        options.setdefault("yanchor", yanchor)
    return plotly.graph_objects.layout.scene.Annotation(
            text=text, x=x, y=y, z=z, **options)


def points3d(points, **options):
    options = options.copy()
    x, y, z = np.array(points, dtype=float).transpose()
    options.setdefault("marker_color", options.pop("color", None))
    options.setdefault("marker_size", options.pop("size", 2.5))
    options.setdefault("mode", "markers")
    return plotly.graph_objects.Scatter3d(x=x, y=y, z=z, **options)


def lines3d(points, **options):
    options = options.copy()
    x, y, z = np.array(points, dtype=float).transpose()
    options.setdefault("line_color", options.pop("color", None))
    options.setdefault("mode", "lines")
    return plotly.graph_objects.Scatter3d(x=x, y=y, z=z, **options)


def function3d(f, x_range, y_range, **options):
    options = options.copy()
    samples = options.pop("samples", 81)
    try:
        samplesx, samplesy = samples
    except:
        samplesx = samplesy = samples
    if (color := options.pop("color", None)) is not None:
        options.setdefault("colorscale", (color, color))
    xmin, xmax = x_range
    ymin, ymax = y_range
    f_ = vectorize(f, 2, 1)
    x = np.linspace(xmin, xmax, samplesx)
    y = np.linspace(ymin, ymax, samplesy)
    z = f_(*np.meshgrid(x, y))
    return plotly.graph_objects.Surface(x=x, y=y, z=z, **options)


def parametric_curve3d(f, t_range, **options):
    options = options.copy()
    plotpoints = options.pop("plotpoints", 101)
    options.setdefault("line_color", options.pop("color", "blue"))
    options.setdefault("mode", "lines")
    t, tmin, tmax = t_range
    f1, f2, f3 = [fast_float(f_, t) for f_ in f]
    t = np.linspace(float(tmin), float(tmax), plotpoints)
    x, y, z = np.array([(f1(t0), f2(t0), f3(t0)) for t0 in t]).transpose()
    if options.pop("update", False):
        return dict(x=x, y=y, z=z)
    return plotly.graph_objects.Scatter3d(x=x, y=y, z=z, **options)


def parametric_surface3d(f, u_range, v_range, **options):
    samples = options.pop("samples", 81)
    try:
        samplesu, samplesv = samples
    except:
        samplesu = samplesv = samples
    if (color := options.pop("color", None)) is not None:
        options.setdefault("colorscale", (color, color))
    umin, umax = u_range
    vmin, vmax = v_range
    f_ = vectorize(f, 2, 3)
    u = np.linspace(umin, umax, samplesu)
    v = np.linspace(vmin, vmax, samplesv)
    u, v = np.meshgrid(u, v)
    x, y, z = f_(u, v)
    return plotly.graph_objects.Surface(x=x, y=y, z=z, **options)


def implicit3d(f, x_range, y_range, z_range, **options):
    options = options.copy()
    plotpoints = options.pop("plotpoints", 41)
    try:
        plotpointsx, plotpointsy, plotpointsz = plotpoints
    except:
        plotpointsx = plotpointsy = plotpointsz = plotpoints
    color=options.pop("color", "lightblue")
    options.setdefault("colorscale", [[0, color], [1, color]])
    options.setdefault("isomin", 0)
    options.setdefault("isomax", 0)
    options.setdefault("surface_count", int(1))
    x, xmin, xmax = x_range
    y, ymin, ymax = y_range
    z, zmin, zmax = z_range
    f = fast_float(f, x, y, z)
    x = np.linspace(float(xmin), float(xmax), plotpointsx)
    y = np.linspace(float(ymin), float(ymax), plotpointsy)
    z = np.linspace(float(zmin), float(zmax), plotpointsz)
    x, y, z = [a.flatten() for a in np.meshgrid(x, y, z)]
    value = np.array([f(x0, y0, z0) for x0, y0, z0 in zip(x, y, z)])
    if options.pop("update", False):
        return dict(x=x, y=y, z=z, value=value)
    return plotly.graph_objects.Isosurface(x=x, y=y, z=z, value=value, **options)


def vector3d(vec, start=(0, 0, 0), **options):
    options = options.copy()
    tipsize = options.pop("tipsize", 0.20)
    shaftwidth = options.pop("shaftwidth", 3)
    color = options.pop("color", "black")
    colorscale = options.pop("colorscale", (color, color))
    options.setdefault("legendgroup", f"{randrange(1 << 32):08X}")

    options["mode"] = "lines"
    options.setdefault("line_color", color)
    options.setdefault("line_width", shaftwidth)
    start = np.array(start, dtype=float)
    vec = np.array(vec, dtype=float)
    x1, y1, z1 = start
    x2, y2, z2 = start + (1 - tipsize) * vec
    u, v, w = tipsize * vec
    if options.pop("update", False):
        return dict(x=[x1, x2], y=[y1, y2], z=[z1, z2]), dict(x=[x2], y=[y2], z=[z2], u=[u], v=[v], w=[w])
    shaft = plotly.graph_objects.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2], **options)
    del options["mode"]
    del options["line_color"]
    del options["line_width"]

    options["colorscale"] = colorscale
    options["showlegend"] = False
    options["anchor"] = "tail"
    options["sizemode"] = "scaled"
    options["sizeref"] = 1
    tip = plotly.graph_objects.Cone(x=[x2], y=[y2], z=[z2], u=[u], v=[v], w=[w], **options)
    return shaft, tip


def vector_field3d(f, x_range, y_range, z_range, **options):
    options = options.copy()
    plotpoints = options.pop("plotpoints", 21)
    try:
        plotpointsx, plotpointsy, plotpointsz = plotpoints
    except:
        plotpointsx = plotpointsy = plotpointsz = plotpoints
    color = options.pop("color", "limegreen")
    options.setdefault("colorscale", [[0, color], [1, color]])
    x, xmin, xmax = x_range
    y, ymin, ymax = y_range
    z, zmin, zmax = z_range
    #diagonal = sqrt( ((xmax - xmin)^2/plotpointsx^2 + (ymax - ymin)^2/plotpointsy^2) + (zmax - zmin)^2/plotpointsz^2) )
    f1, f2, f3 = [fast_float(f_, x, y, z) for f_ in f]
    x = np.linspace(float(xmin), float(xmax), plotpointsx)
    y = np.linspace(float(ymin), float(ymax), plotpointsy)
    z = np.linspace(float(zmin), float(zmax), plotpointsz)
    xyz = np.array(np.meshgrid(x, y, z)).reshape(3, -1)
    uvw = np.array([(f1(x0, y0, z0), f2(x0, y0, z0), f3(x0, y0, z0)) for x0, y0, z0 in xyz.transpose()]).transpose()
    x, y, z = xyz
    u, v, w = uvw
    #scale = diagonal / np.linalg.norm(uvw, axis=0).max()
    #options.setdefault("scale", scale)
    if options.pop("update", False):
        return dict(x=x, y=y, z=z, u=u, v=v, w=w)
    return plotly.graph_objects.Cone(x=x, y=y, z=z, u=u, v=v, w=w, **options)


def trajectory3d(trajectory, **options):
    options = options.copy
    color = options.pop("color", "blue")
    arrows = options.pop("arrows", 6)
    options.setdefault("legendgroup", f"{randrange(1 << 32):08X}")
    options.setdefault("mode", "lines")
    options.setdefault("line_color", color)
    x, y, z = trajectory.transpose()
    update = options.pop("update", False)
    if update:
        p1 = dict(x=x, y=y, z=z)
    else:
        p1 = plotly.graph_objects.Scatter3d(x=x, y=y, z=z, **options)

    del options["mode"]
    options["showlegend"] = False
    options.setdefault("colorscale", (color, color))
    options.setdefault("sizeref", 0.33) # Adjust this for smaller/largers cones
    try:
        arrows = np.array(list(arrows), dtype=int)
    except:
        arclength = np.cumsum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1))
        arrows_so_far = np.rint(arrows*arclength / arclength[-1])
        arrows = np.where(arrows_so_far[1:] - arrows_so_far[:-1])[0]
    cone_pos = trajectory[arrows]
    cone_dir = trajectory[arrows+1] - cone_pos
    cone_dir /= np.linalg.norm(cone_dir, axis=1)[:,np.newaxis]
    x, y, z = cone_pos.transpose()
    u, v, w = cone_dir.transpose()
    if update:
        p1 = dict(x=x, y=y, z=z, u=u, v=v, w=w)
    else:
        p2 = plotly.graph_objects.Cone(x=x, y=y, z=z, u=u, v=v, w=w, **options)
    return (p1, p2)


def vector_field_on_surface3d(f, field, u_range, v_range, **options):
    options = options.copy()
    plotpoints = options.pop("plotpoints", 21)
    try:
        plotpointsu, plotpointsv = plotpoints
    except:
        plotpointsu = plotpointsv = plotpoints
    color = options.pop("color", "limegreen")
    options.setdefault("colorscale", (color, color))
    u, umin, umax = u_range
    v, vmin, vmax = v_range
    f1, f2, f3 = [fast_float(f_, u, v) for f_ in f]
    if len(field) == 2:
        field = f.derivative() * field
    fx, fy, fz = [fast_float(f_, u, v) for f_ in field]
    u = np.linspace(float(umin), float(umax), plotpointsu)
    v = np.linspace(float(vmin), float(vmax), plotpointsv)
    uv = np.array(np.meshgrid(u, v)).reshape(2, -1).transpose()
    points = np.array([(f1(u0, v0), f2(u0, v0), f3(u0, v0)) for u0, v0 in uv])
    vectors = np.array([(fx(u0, v0), fy(u0, v0), fz(u0, v0)) for u0, v0 in uv])
    x, y, z = points.transpose()
    u, v, w = vectors.transpose()
    if options.pop("update", False):
        return dict(x=x, y=y, z=z, u=u, v=v, w=w)
    return plotly.graph_objects.Cone(x=x, y=y, z=z, u=u, v=v, w=w, **options)


#def surface_intersection(surface, u, v, u0, v0, normal, tmin=-10, tmax=10, tstep=0.01, **options):
#    color=options.pop("color", "blue")
#    T_u = diff(surface, u)
#    T_v = diff(surface, v)
#    norm = sqrt((T_u * T_u) * (T_v * T_v) - (T_u * T_v)^2)
#    vf(u, v) = (T_v(u, v) * normal / norm(u, v), -T_u(u, v) * normal / norm(u, v))
#    solution1 = desolve_odeint(vf, (u0, v0), srange(0, tmax, tstep), [u, v])
#    solution2 = desolve_odeint(-vf, (u0, v0), srange(0, -tmin, tstep), [u, v])
#    solution = np.concatenate((np.flip(solution2[1:], axis=0), solution1))
#    f1, f2, f3 = [fast_float(f_, u, v) for f_ in surface]
#    x, y, z = np.array([(f1(u0, v0), f2(u0, v0), f3(u0, v0)) for u0, v0 in solution]).transpose()
#    return plotly.graph_objects.Scatter3d(x=x, y=y, z=z, line_color=color, **options)

