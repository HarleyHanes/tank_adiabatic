import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def subplotMovie(yVariables, xVariables, output_filename, fps=5, xLabels="X", yLabels="null", legends="null",legendLoc="upper left",subplotSize=(5, 4),yRanges="auto"):
    """
    Create a .mov file where each frame is a subplot call using the ith element
    in the first dimension of every numpy array in yVariables. Returns the video_writer.

    Parameters:
    - xVariables: List of x-values for the plots.
    - yVariables: List of y-values (2D+ arrays, first dimension is the frame index).
    - output_filename: Output .mov file name.
    - fps: Frames per second for the video.
    - xLabels: X-axis labels for the subplots.
    - yLabels: Y-axis labels for the subplots.
    - legends: Legends for plots
    - subplotSize: Size of each subplot in inches (width, height).

    Returns:
    - video_writer: The cv2.VideoWriter object.
    """
    # Check if yVariables are compatible
    num_frames = yVariables[0].shape[0]
    for yVar in yVariables:
        if yVar.shape[0] != num_frames:
            raise ValueError("All yVariables must have the same number of frames.")

    # Validate yRanges
    if yRanges == "auto":
        yRanges = []
        for y in yVariables:
            y_min, y_max = np.min(y), np.max(y)
            padding = 0.05 * (y_max - y_min)
            yRanges.append((y_min - padding, y_max + padding))
    elif isinstance(yRanges, list):
        if len(yRanges) != len(yVariables):
            raise ValueError(f"Invalid length of {len(yRanges)} for yRanges for yVariables of length {len(yVariables)}")
    else:
        raise ValueError(f"Invalid type entered for yRanges: {type(yRanges)}")


    # Matplotlib figure size (in pixels)
    dpi = 400
    # Determine grid dimensions where rows * cols = numPlots and rows >= cols
    numPlots = len(yVariables)
    cols = int(math.sqrt(numPlots))
    while numPlots % cols != 0:
        cols -= 1
    rows = numPlots // cols

    # Calculate the figure size based on the number of rows, columns, and subplot size
    figWidth, figHeight = subplotSize[0] * cols* dpi, subplotSize[1] * rows* dpi
    frame_size = (int(figWidth), int(figHeight))  # OpenCV expects (width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec compatible with .mov
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    # Loop through each frame
    for i in range(num_frames):
        # Prepare current frame's yVariables
        frame_yVariables = [yVar[i, :] for yVar in yVariables]

        # Create the subplot using the provided function
        fig, axes = subplot(frame_yVariables, xVariables, xLabels=xLabels, yLabels=yLabels,legends=legends,legendLoc=legendLoc, subplotSize=subplotSize,yRanges=yRanges)

        # Render the figure to a buffer
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize the frame to match the expected resolution
        img_bgr = cv2.resize(img_bgr, frame_size)

        # Write frame to video
        video_writer.write(img_bgr)

        # Close the figure to free resources
        plt.close(fig)
    video_writer.release()

def subplot(yVariables, xVariables, xLabels="X", yLabels="Y", legends="null", legendLoc="best", subplotSize=(5, 4),yRanges="auto"):
    # Validate inputs
    if isinstance(yVariables, np.ndarray) and isinstance(xVariables, np.ndarray):
        if yVariables.shape[-1] != xVariables.shape[-1]:
            raise ValueError("The last dimensions of yVariables and xVariables must be equal.")
        yVariables, xVariables = [yVariables], [xVariables]

    elif isinstance(yVariables, list) and isinstance(xVariables, np.ndarray):
        if not all(isinstance(y, np.ndarray) for y in yVariables):
            raise ValueError("All elements in yVariables must be numpy arrays.")
        if not all(y.shape[-1] == xVariables.shape[-1] for y in yVariables):
            errorString = "Each numpy array in yVariables must have the same last dimension as xVariables.\ny Shapes: "
            for y in yVariables:
                errorString +=str(y.shape)+", "
            errorString+= "\nx Shape: " + str(xVariables.shape)
            raise ValueError(errorString)
        xVariables = [xVariables] * len(yVariables)

    elif isinstance(yVariables, list) and isinstance(xVariables, list):
        if len(yVariables) != len(xVariables):
            raise ValueError("yVariables and xVariables must have the same number of elements.")
        if not all(isinstance(y, np.ndarray) and isinstance(x, np.ndarray) for y, x in zip(yVariables, xVariables)):
            raise ValueError("All elements in yVariables and xVariables must be numpy arrays.")
        if not all(y.shape[-1] == x.shape[-1] for y, x in zip(yVariables, xVariables)):
            raise ValueError("Each pair of corresponding arrays in yVariables and xVariables must have equal last dimension sizes.")
        
    if isinstance(xLabels, list): 
        if len(xLabels)!=len(xVariables):
            raise ValueError("Invalid length of "+ str(len(xLabels))+" for xLabels for xVariables of length " + len(xVariables))
    elif isinstance(xLabels,str):
        xLabels = [xLabels] * len(xVariables)
    else:
        raise ValueError("Invalid type entered for xLabels: " + str(type(xLabels)))
    
    # Determine if xLabels and yLabels are lists
    useIndividualXLabels = isinstance(xLabels, list)
    useIndividualYLabels = isinstance(yLabels, list)

    if isinstance(yLabels, list):
        if len(yLabels)!=len(yVariables):
            raise ValueError("Invalid length of "+ str(len(yLabels))+" for yLabels for yVariables of length " + len(yVariables))
    elif isinstance(yLabels,str):
        yLabels = [yLabels] * len(yVariables)
    else:
        raise ValueError("Invalid type entered for yLabels: " + str(type(yLabels)))
    if legends=="null":
        useLegends=False
    else:
        useLegends=True
        if all(isinstance(legend, list) for legend in legends):
            if len(legends)!=len(yVariables):
                raise ValueError("Invalid length of "+ str(len(legends))+" for legends for yVariables of length " + len(yVariables))
            for i in range(len(yVariables)):
                if len(legends[i])!=yVariables[i].shape[0]:
                    raise ValueError("Legend " + str(i) + "is of length " + str(len(legends[i]))+ " but yVariables has "+ str(yVariables[i].shape[0])+" lines.")
                if not all(isinstance(lineLabels,str) for lineLabels in legends[i]):
                    raise ValueError("Non-string legend elements for legend[" + str(i)+"]")
            useIndividualLegends=True
        elif all(isinstance(legend, str) for legend in legends):
                if len(legends)!=yVariables[0].shape[0]:
                    print(legends)
                    raise ValueError("String legend is of length " + str(len(legends))+ " but yVariables has "+ str(yVariables[0].shape[0])+" lines.")
                useIndividualLegends=False
        else:
            raise ValueError("Invalid types entered for legends. Must be a list of strings or list of list of strings: " + str(type(legends[0])))
    
    # Validate yRanges
    if yRanges == "auto":
        yRanges = []
        for y in yVariables:
            y_min, y_max = np.min(y), np.max(y)
            padding = 0.05 * (y_max - y_min)
            yRanges.append((y_min - padding, y_max + padding))
    elif isinstance(yRanges, list):
        if len(yRanges) != len(yVariables):
            raise ValueError(f"Invalid length of {len(yRanges)} for yRanges for yVariables of length {len(yVariables)}")
    else:
        raise ValueError(f"Invalid type entered for yRanges: {type(yRanges)}")

    # Determine grid dimensions where rows * cols = numPlots and rows >= cols
    numPlots = len(yVariables)
    cols = int(math.sqrt(numPlots))
    while numPlots % cols != 0:
        cols -= 1
    rows = numPlots // cols

    # Calculate the figure size based on the number of rows, columns, and subplot size
    figWidth, figHeight = subplotSize[0] * cols, subplotSize[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(figWidth, figHeight))
    axes = np.atleast_2d(axes).flatten()  # Flatten to easily index each subplot


    # Plot each pair of (y, x) in subplots with conditional x-axis and y-axis labeling
    for i, (y, x) in enumerate(zip(yVariables, xVariables)):
        for iline in range(y.shape[0]):
            axes[i].plot(x, y[iline],getLineFormat("line",iline),lw=subplotSize[0],ms=2*subplotSize[0])
        axes[i].set_ylim(yRanges[i])  # Set the y-axis range

        # Apply x-axis labels according to the conditions
        if useIndividualXLabels:
            axes[i].set_xlabel(xLabels[i])
        elif i >= (rows - 1) * cols:  # Only label x-axis for the last row
            axes[i].set_xlabel(xLabels[0])

        if useIndividualYLabels:
            axes[i].set_ylabel(yLabels[i],rotation=0,labelpad=10.0)
        elif i % cols == 0:  # Only label y-axis for the left-most column
            axes[i].set_ylabel(yLabels[0],rotation=0,labelpad=10.0)
        if useLegends:
            if useIndividualLegends:
                axes[i].legend(legends[i], loc = legendLoc)
            elif i==cols-1:
                axes[i].legend(legends,  loc = legendLoc)

    plt.tight_layout()
    return fig, axes

def subplotTimeSeries(yVariables, xVariables, xLabels="X", yLabels="Y", title="null", legends="null", legendLoc="best",subplotSize=(5, 4)):
    # Validate inputs
    if isinstance(yVariables, np.ndarray) and isinstance(xVariables, np.ndarray):
        if yVariables.shape[-1] != xVariables.shape[-1]:
            raise ValueError("The last dimensions of yVariables and xVariables must be equal.")
        yVariables, xVariables = [yVariables], [xVariables]

    elif isinstance(yVariables, list) and isinstance(xVariables, np.ndarray):
        if not all(isinstance(y, np.ndarray) for y in yVariables):
            raise ValueError("All elements in yVariables must be numpy arrays.")
        if not all(y.shape[-1] == xVariables.shape[-1] for y in yVariables):
            errorString = "Each numpy array in yVariables must have the same last dimension as xVariables.\ny Shapes: "
            for y in yVariables:
                errorString +=str(y.shape)+", "
            errorString+= "\nx Shape: " + str(xVariables.shape)
            raise ValueError(errorString)
        xVariables = [xVariables] * len(yVariables)

    elif isinstance(yVariables, list) and isinstance(xVariables, list):
        if len(yVariables) != len(xVariables):
            raise ValueError("yVariables and xVariables must have the same number of elements.")
        if not all(isinstance(y, np.ndarray) and isinstance(x, np.ndarray) for y, x in zip(yVariables, xVariables)):
            raise ValueError("All elements in yVariables and xVariables must be numpy arrays.")
        if not all(y.shape[-1] == x.shape[-1] for y, x in zip(yVariables, xVariables)):
            raise ValueError("Each pair of corresponding arrays in yVariables and xVariables must have equal last dimension sizes.")

    if isinstance(yVariables,list) and yVariables[0].ndim>1:
        if not all(y.shape[0]==yVariables[0].shape[0] for y in yVariables):
            raise ValueError("All elements of yVariables must have same first dimnsion if at least 2 dimensions")
    #Resize to have a leading variable of 1 to work with column indexing
    elif isinstance(yVariables,list) and yVariables[0].ndim==1:
        for i in len(yVariables):
            yVariables[i]=np.reshape(yVariables[i],(1,)+yVariables[i].shape) 
    if isinstance(xLabels, list): 
        if len(xLabels)!=len(xVariables):
            raise ValueError("Invalid length of "+ str(len(xLabels))+" for xLabels for xVariables of length " + len(xVariables))
    elif isinstance(xLabels,str):
        xLabels = [xLabels] * len(xVariables)
    else:
        raise ValueError("Invalid type entered for xLabels: " + str(type(xLabels)))
    
    # Determine if xLabels and yLabels are lists
    useIndividualXLabels = isinstance(xLabels, list)
    useIndividualYLabels = isinstance(yLabels, list)
    useIndividualTitles = isinstance(title, list)

    if isinstance(yLabels, list):
        if len(yLabels)!=len(yVariables):
            raise ValueError("Invalid length of "+ str(len(yLabels))+" for yLabels for yVariables of length " + len(yVariables))
    elif isinstance(yLabels,str):
        yLabels = [yLabels] * len(yVariables)
    else:
        raise ValueError("Invalid type entered for yLabels: " + str(type(yLabels)))
    
    if legends=="null":
        useLegends=False
    else:
        useLegends=True
        if all(isinstance(legend, list) for legend in legends):
            if len(legends)!=len(yVariables):
                raise ValueError("Invalid length of "+ str(len(legends))+" for legends for yVariables of length " + len(yVariables))
            for i in range(len(yVariables)):
                if len(legends[i])!=yVariables[i].shape[-2]:
                    raise ValueError("Legend " + str(i) + "is of length " + str(len(legends[i]))+ " but yVariables has "+ str(yVariables[i].shape[-2])+" lines.")
                if not all(isinstance(lineLabels,str) for lineLabels in legends[i]):
                    raise ValueError("Non-string legend elements for legend[" + str(i)+"]")
            useIndividualLegends=True
        elif all(isinstance(legend, str) for legend in legends):
                if len(legends)!=yVariables[0].shape[-2]:
                    print(legends)
                    raise ValueError("String legend is of length " + str(len(legends))+ " but yVariables has "+ str(yVariables[0].shape[-2])+" lines.")
                useIndividualLegends=False
        else:
            raise ValueError("Invalid types entered for legends. Must be a list of strings or list of list of strings: " + str(type(legends[0])))
    
    # Determine grid dimensions where rows * cols = numPlots and rows >= cols
    rows = len(yVariables)
    cols = yVariables[0].shape[0]
   

    # Calculate the figure size based on the number of rows, columns, and subplot size
    figWidth, figHeight = subplotSize[0] * cols, subplotSize[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(figWidth, figHeight))
    axes = np.reshape(axes,(rows,cols))

    # Plot each pair of (y, x) in subplots with conditional x-axis and y-axis labeling
    for iy in range(len(yVariables)):
        ywidth = np.max(yVariables[iy])-np.min(yVariables[iy])
        yRanges=(np.min(yVariables[iy])-.1*ywidth,np.max(yVariables[iy])+.1*ywidth)
        for it in range(yVariables[0].shape[0]):
            if len(xVariables)==len(yVariables):
                for iline in range(yVariables[iy][it].shape[0]):
                    axes[iy,it].plot(xVariables[iy], yVariables[iy][it][iline],getLineFormat("line",iline),lw=subplotSize[0],ms=2*subplotSize[0])
            else:
                for iline in range(yVariables[iy][it].shape[0]):
                    axes[iy,it].plot(xVariables[0], yVariables[iy][it][iline],getLineFormat("line",iline),lw=subplotSize[0],ms=2*subplotSize[0])
            axes[iy,it].set_ylim(yRanges)
            if useIndividualTitles and title!="null" and iy==0:
                axes[iy, it].set_title(title[it])
            elif not useIndividualTitles and title!="null":
                fig.suptitle(title, fontsize=16)

            # Apply x-axis labels according to the conditions
            if useIndividualXLabels:
                axes[iy, it].set_xlabel(xLabels[iy])
            elif i >= (rows - 1) * cols:  # Only label x-axis for the last row
                axes[iy, it].set_xlabel(xLabels[0])

            # Apply y-axis labels according to the conditions
            if it == 0:
                if useIndividualYLabels:
                    axes[iy, it].set_ylabel(yLabels[iy],rotation=0,labelpad=8.0)
                else:  # Only label y-axis for the left-most column
                    axes[iy, it].set_ylabel(yLabels[0],rotation=0,labelpad=8.0)
            if useLegends and it ==yVariables[0].shape[0]-1:
                if useIndividualLegends:
                    axes[iy, it].legend(legends[iy], loc = legendLoc)
                elif iy==0:
                    axes[iy, it].legend(legends,  loc = legendLoc)

    plt.tight_layout()
    return fig, axes

def getLineFormat(linetype,iter):
    if linetype == "line":
        match iter:
            case 0:
                return "-b"
            case 1:
                return "--m"
            case 2:
                return "-.g"
            case 3:
                return "-*r"
    elif linetype == "point":
        match iter:
            case 0:
                return "sb"
            case 1:
                return "tm"
            case 2:
                return ".g"
            case 3:
                return "*r"
    else :
        raise ValueError("Invalid linetype option, must be line or point")