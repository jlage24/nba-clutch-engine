import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle


def draw_court(ax=None, color="black", lw=2):
    # NBA half-court in LOC_X / LOC_Y units (tenths of a foot, hoop at 0,0).
    ax = ax or plt.gca()
    elements = [
        Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False),                 # hoop
        Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color),                         # backboard
        Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False),          # outer paint
        Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False),          # inner paint
        Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color),        # free-throw arc (top)
        Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color,
            linestyle="dashed"),                                                           # free-throw arc (bottom)
        Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color),              # restricted area
        Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color),                       # left corner three
        Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color),                        # right corner three
        Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color),           # three-point arc
        Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False),         # boundary
    ]
    for e in elements:
        ax.add_patch(e)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax