'''
Not commented because it's not directly relevant to the real functionality of the
code base.
'''
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3


def vis_3d(visualize_gt, how_many_objects):
    fig = plt.figure()
    ax = p3.Axes3D(auto_add_to_figure=False, fig=fig)
    fig.add_axes(ax)

    # ax1 = p3.Axes3D(fig)

    def update(num, line2, rdf, scatter):
        ax.clear()

        # Setting the axes properties
        ax.set_xlim3d([-4.3, 4.61])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.66, 4.73])
        ax.set_ylabel('Z')  # change the label to z since z-axis values are given previously

        ax.set_zlim3d([-1.0, 4.3])
        ax.set_zlabel('Y')  # change the label to y since y-axis values are given previously
        how_many_objs = 10
        olist = []
        lendf = len(rdf)
        if how_many_objects > 0 and lendf > 0:
            t_tdf1 = rdf[0]
            ax.scatter(t_tdf1[0, num], t_tdf1[1, num], t_tdf1[2, num], marker=".", color='b')


        ax.scatter(0.0, 0.0, 0.0, marker="D")
        ax.scatter(2.0, 0.0, 0.0, marker="D")
        # t_l1.set_3d_properties(v)
        # t_l2.set_3d_properties(t_tdf2[2, :num])
        # t_l3.set_3d_properties(t_tdf3[2, :num])
        # t_l4.set_3d_properties(t_tdf4[2, :num])
        # t_l5.set_3d_properties(t_tdf5[2, :num])
        # t_l6.set_3d_properties(t_tdf6[2, :num])
        # t_l7.set_3d_properties(t_tdf7[2, :num])
        # t_l8.set_3d_properties(t_tdf8[2, :num])
        # t_l9.set_3d_properties(t_tdf9[2, :num])
        # t_l10.set_3d_properties(t_tdf10[2, :num])

    lines = []
    scatter, = ax.plot([], [], [], 'b', animated=True)
    dfnp2 = []
    if visualize_gt:
        df_recon = pd.read_csv("gt_files/gt_3d_data_v1.csv")
    else:
        df_recon = pd.read_csv("result_files/reconstruction.csv")
    N = len(df_recon)
    unique_ids = df_recon.gtoid.unique()
    for i in range(len(unique_ids)):
        df_recon_2 = df_recon.loc[df_recon['gtoid'] == int(unique_ids[i])]
        df_recon_2 = df_recon_2[0:]
        dfnp = np.zeros((3, len(df_recon_2)))
        dfnp[0] = df_recon_2.iloc[0:]['x']
        dfnp[2] = df_recon_2.iloc[0:]['y']  # z value is passed here
        dfnp[1] = df_recon_2.iloc[0:]['z']  # y value is passed here
        l1, = ax.plot(dfnp[0, 0:1], dfnp[1, 0:1], dfnp[2, 0:1])
        lines.append(l1)
        dfnp2.append(dfnp)

        '''
        df_recon = df_recon.loc[df_recon['oid'] == 0]
        N = len(df_recon)
        dfnp = np.zeros((3, len(df_recon)))
        dfnp[0] = df_recon.iloc[0:]['x']
        dfnp[1] = df_recon.iloc[0:]['y']
        dfnp[2] = df_recon.iloc[0:]['z']
        line2, = ax.plot(dfnp[0, 0:1], dfnp[1, 0:1], dfnp[2, 0:1])
        '''

    ani = animation.FuncAnimation(fig, update, N, fargs=(lines, dfnp2, scatter), interval=100,
                                  blit=False)
    #  ani = animation.FuncAnimation(fig, update, N, fargs=(line2, dfnp, len(df_recon)), interval=10000 / N,
    #                              blit=False)
    # ani.save('matplot003.gif', writer='imagemagick')
    plt.show()
    # ani.save(f'result_files/objects_trajectories.gif', writer='imagemagick', fps=30)
    FFwriter = animation.FFMpegWriter()
    ani.save(f'result_files/objects_trajectories.mp4', writer="ffmpeg", bitrate=1500, fps=15)

def vis_2d(visualize_gt, fname, which_camera, no_geo):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)

    def animate(i, l1, l2):
        # line.set_xdata(dfs[0, :i])
        # line.set_ydata(dfs[1, :i])
        ax.clear()
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)

        if i > 10:
            line1 = ax.plot(l1[0, i - 10:i], l1[1, i - 10:i], c="r", marker=".")
            line2 = ax.plot(l2[0, i - 10:i], l2[1, i - 10:i], c="b", marker=".")

        else:
            line1 = ax.plot(l1[0, :i], l1[1, :i], c="r", marker=".")
            line2 = ax.plot(l2[0, :i], l2[1, :i], c="b", marker=".")

        plt.text(50, 1000, f'{i}')
        return line1, line2

    if visualize_gt:
        df_recon = pd.read_csv("gt_files/proj_2d.csv")
        dfs = []
        for i in range(2):
            df_recon_2 = df_recon.loc[df_recon['cid'] == which_camera.id]

            df_recon_2 = df_recon_2.loc[df_recon['oid'] == i]
            dfnp = np.zeros((2, len(df_recon_2)))
            dfnp[0] = df_recon_2.iloc[0:]['x']
            dfnp[1] = df_recon_2.iloc[0:]['y']
            dfs.append(dfnp)
    else:
        if no_geo:
            df_recon = pd.read_csv("result_files/trajectories2d_no_geo.csv")
        else:
            df_recon = pd.read_csv("result_files/trajectories2d.csv")
        dfs = []
        for i in range(2):
            df_recon_2 = df_recon.sort_values('frame')
            df_recon_2 = df_recon_2.loc[df_recon_2['tid'] == i]
            dfnp = np.zeros((2, len(df_recon_2)))
            dfnp[0] = df_recon_2.iloc[0:]['x']
            dfnp[1] = df_recon_2.iloc[0:]['y']
            dfs.append(dfnp)

    ani = animation.FuncAnimation(
        fig, animate, frames=599, fargs=dfs, interval=33, blit=False,
        repeat=True, save_count=599)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    # writer = animation.FFMpegWriter(fps=5)
    # ani.save(f'result_files/{fname}.gif', writer=writer)
    ani.save(f'result_files/{fname}.gif', writer='imagemagick', fps=5)


def vis_2d_tid(visualize_gt, fname, which_camera, tid_1, tid_2):
    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i, l1, l2):
        # line.set_xdata(dfs[0, :i])
        # line.set_ydata(dfs[1, :i])
        ax.clear()
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        line1 = ax.plot(l1[0, :i], l1[1, :i], c="r", marker=".")
        line2 = ax.plot(l2[0, :i], l2[1, :i], c="b", marker=".")

        plt.text(50, 1000, f'{i}')
        return line1, line2

    df_recon = pd.read_csv("result_files/trj_other_camera.csv")
    dfs = []

    df_recon_2 = df_recon.sort_values('frame')
    df_recon_2 = df_recon_2.loc[df_recon_2['tid'] == tid_1]
    dfnp = np.zeros((2, len(df_recon_2)))
    dfnp[0] = df_recon_2.iloc[0:]['x']
    dfnp[1] = df_recon_2.iloc[0:]['y']
    dfs.append(dfnp)

    df_recon = pd.read_csv("result_files/trajectories2d.csv")
    df_recon_2 = df_recon.sort_values('frame')
    df_recon_2 = df_recon_2.loc[df_recon_2['tid'] == tid_2]
    dfnp = np.zeros((2, len(df_recon_2)))
    dfnp[0] = df_recon_2.iloc[0:]['x']
    dfnp[1] = df_recon_2.iloc[0:]['y']
    dfs.append(dfnp)

    ani = animation.FuncAnimation(
        fig, animate, frames=200, fargs=dfs, interval=33, blit=False,
        repeat=True, save_count=200)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    writer = animation.FFMpegWriter(fps=30)
    # plt.show()
    ani.save(f'result_files/{fname}.mp4', writer=writer)
    #  ani.save(f'result_files/{fname}.mp4

def vis_3d_scatter(list_of_points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    c = 0
    for p in list_of_points:
        if c < 2:
            ax.scatter(p[0], p[1], p[2], marker="D")
        else:
            ax.scatter(p[0], p[1], p[2], marker=".")
        c += 1

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()