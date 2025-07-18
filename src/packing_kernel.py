"""
Packing Engine: Basic Classes for the Bin Packing Problem
We follow the space representation depicted below, all coordinates and lengths of boxes and containers are integers.

    x: depth
    y: length
    z: height

       Z
       |
       |
       |________Y
      /
     /
    X

    Classes:
        Box
        Container

"""
from copy import deepcopy
from typing import List, Type

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nptyping import NDArray, Int, Shape

from utils import (
    generate_vertices,
    boxes_generator,
    cuboids_intersection,
    cuboid_fits,
)


class Box:
    """A class to represent a 3D box

    Attributes
    ----------
     id_: int
           id of the box
     position: int
           Coordinates of the position of the bottom-leftmost-deepest corner of the box
     size: int
           Lengths of the edges of the box
    """

    def __init__(self, size: List[int], position: List[int], id_: int) -> None:
        """Initializes a box object

        Parameters
        ----------
        size: List[int]
            Lengths of the edges of the box in the order (x, y, z) = (depth, length, height)
        position: List[int]
            Coordinates of the position of the bottom-leftmost-deepest corner of the box
        id_: int
            id of the box

        Returns
        -------
        Box object
        """
        assert len(size) == len(
            position
        ), "Lengths of box size and position do not match"
        assert len(size) == 3, "Box size must be a list of 3 integers"

        assert (
            size[0] > 0 and size[1] > 0 and size[2] > 0
        ), "Lengths of edges must be positive"
        assert (position[0] == -1 and position[1] == -1 and position[2] == -1) or (
            position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        ), "Position is not valid"

        self.id_ = id_
        self.position = np.asarray(position)
        self.size = np.asarray(size)

    def rotate(self, rotation: int) -> None:
        """Rotates the box in place

        Parameters
        ----------
        rotation: int
        """
        pass  # to be added later

    @property
    def area_bottom(self) -> int:
        """Area of the bottom face of the box"""
        return self.size[0] * self.size[1]

    @property
    def volume(self) -> int:
        """Volume of the box"""
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def vertices(self) -> NDArray:
        """Returns a list with the vertices of the box"""
        vert = generate_vertices(self.size, self.position)
        return np.asarray(vert, dtype=np.int32)

    def __repr__(self):
        return (
            f"Box id: {self.id_}, Size: {self.size[0]} x {self.size[1]} x {self.size[2]}, "
            f"Position: ({self.position[0]}, {self.position[1]}, {self.position[2]})"
        )

    def plot(self, color, figure: Type[go.Figure] = None) -> Type[go.Figure]:
        """Adds the plot of a box to a given figure

         Parameters
         ----------
        figure: go.Figure
             A plotly figure where the box should be plotted

         Returns
         -------
         go.Figure
        """
        # Generate the coordinates of the vertices
        vertices = generate_vertices(self.size, self.position).T
        x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
        # The arrays i, j, k contain the indices of the triangles to be plotted (two per each face of the box)
        # The triangles have vertices (x[i[index]], y[j[index]], z[k[index]]), index = 0,1,..7.
        i = [1, 2, 5, 6, 1, 4, 3, 6, 1, 7, 0, 6]
        j = [0, 3, 4, 7, 0, 5, 2, 7, 3, 5, 2, 4]
        k = [2, 1, 6, 5, 4, 1, 6, 3, 7, 1, 6, 0]

        edge_pairs = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        for (m, n) in edge_pairs:
            vert_x = np.array([x[m], x[n]])
            vert_y = np.array([y[m], y[n]])
            vert_z = np.array([z[m], z[n]])

        if figure is None:
            # Plot the box faces
            figure = go.Figure(
                data=[
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        opacity=1,
                        color=color,
                        flatshading=True,
                    )
                ]
            )
            # Plot the box edges
            figure.add_trace(
                go.Scatter3d(
                    x=vert_x,
                    y=vert_y,
                    z=vert_z,
                    mode="lines",
                    line=dict(color="black", width=0),
                )
            )

        else:
            # Plot the box faces
            figure.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    opacity=1,
                    color=color,
                    flatshading=True,
                )
            )
            # Plot the box edges
            figure.add_trace(
                go.Scatter3d(
                    x=vert_x,
                    y=vert_y,
                    z=vert_z,
                    mode="lines",
                    line=dict(color="black", width=0),
                )
            )

        return figure


class Container:
    """A class to represent a 3D container

    Attributes
    ----------
    id_: int
        id of the container
    size: NDArray[Shape["1,3"],Int]
        Lengths of the edges of the container
    position: NDArray[Shape["1,3"],Int]
        Coordinates of the bottom-leftmost-deepest corner of the container
    boxes: List[Box]
        List with the boxes placed inside the container
    height_map: NDArray[Shape["*,*"],Int]
        An array of size (size[0],size[1]) representing the height map (top view) of the container,
        where height_map[i,j] is the current height of stacked items at position (i,j).
    """

    def __init__(
        self,
        size: NDArray[Shape["1,3"], Int],
        position: NDArray[Shape["1,3"], Int] = None,
        id_: int = 0,
    ) -> None:
        """Initializes a 3D container

        Parameters
        ----------
        id_: int, optional
            id of the container (default = 0)
        positions: int, optional
            Coordinates of the bottom-leftmost-deepest corner of the container (default = 0,0,0)
        size: int
            Lengths of the edges of the container
        """

        if position is None:
            position = np.zeros(shape=3, dtype=np.int32)

        assert len(size) == len(position), "Sizes of size and position do not match"
        assert len(size) == 3, "Size of size is different from 3"
        position = np.asarray(position)
        np.testing.assert_equal(position[2], 0), "Position is not valid"

        self.id_ = id_
        self.position = np.asarray(position, dtype=np.int32)
        self.size = np.asarray(size, dtype=np.int32)
        self.boxes = []
        self.height_map = np.zeros(shape=(size[0], size[1]), dtype=np.int32)

    @property
    def vertices(self):
        """Returns a list with the vertices of the container"""
        return generate_vertices(self.size, self.position)

    @property
    def volume(self) -> int:
        """Volume of the box"""
        return self.size[0] * self.size[1] * self.size[2]

    def reset(self):
        """Resets the container to an empty state"""
        self.boxes = []
        self.height_map = np.zeros(shape=[self.size[0], self.size[1]], dtype=np.int32)

    def _update_height_map(self, box):
        """Updates the height map after placing a box
         Parameters
        ----------
        box: Box
             Box to be placed inside the container
        """
        # Add the height of the new box in the x-y coordinates occupied by the box
        self.height_map[
            box.position[0] : box.position[0] + box.size[0],
            box.position[1] : box.position[1] + box.size[1],
        ] += box.size[2]

    def __repr__(self):
        return (
            f"Container id: {self.id_}, Size: {self.size[0]} x {self.size[1]} x {self.size[2]}, "
            f"Position: ({self.position[0]}, {self.position[1]}, {self.position[2]})"
        )

    def get_height_map(self):
        """Returns a copy of the height map of the container"""
        return deepcopy(self.height_map)

    def check_valid_box_placement(
        self, box: Box, new_pos: NDArray, check_area: int = 100
    ) -> int:
        """
        Parameters
        ----------
        box: Box
            Box to be placed
        new_pos: NDArray[int]
            Coordinates of new position
        check_area: int, default = 100
             Percentage of area of the bottom of the box that must be supported in the new position

        Returns
        -------
        int
        """
        assert len(new_pos) == 2

        # Generate the vertices of the bottom face of the box
        v = generate_vertices(np.asarray(box.size), np.asarray([*new_pos, 1]))
        # bottom vertices of the box
        v0, v1, v2, v3 = v[0, :], v[1, :], v[2, :], v[3, :]

        # Generate the vertices of the bottom face of the container
        w = generate_vertices(self.size, self.position)
        # bottom vertices of the container
        w0, w1, w2, w3 = w[0, :], w[1, :], w[2, :], w[3, :]

        # Check if the bottom vertices of the box are inside the container
        cond_0 = np.all(np.logical_and(v0[0:2] >= w0[0:2], v0[0:2] <= w3[0:2]))
        cond_1 = np.all(np.logical_and(v1[0:2] >= w0[0:2], v1[0:2] <= w3[0:2]))
        cond_2 = np.all(np.logical_and(v2[0:2] >= w0[0:2], v2[0:2] <= w3[0:2]))
        cond_3 = np.all(np.logical_and(v3[0:2] >= w0[0:2], v3[0:2] <= w3[0:2]))

        # Check if the bottom vertices of the box are inside the container
        if not np.all([cond_0, cond_1, cond_2, cond_3]):
            return 0

        # Check that the bottom vertices of the box in the new position are at the same level
        corners_levels = [
            self.height_map[v0[0], v0[1]],
            self.height_map[v1[0] - 1, v1[1]],
            self.height_map[v2[0], v2[1] - 1],
            self.height_map[v3[0] - 1, v3[1] - 1],
        ]

        if corners_levels.count(corners_levels[0]) != len(corners_levels):
            return 0

        # lev is the level (height) at which the bottom corners of the box will be located
        lev = corners_levels[0]
        # bottom_face_lev contains the levels of all the points in the bottom face
        bottom_face_lev = self.height_map[
            v0[0] : v0[0] + box.size[0], v0[1] : v0[1] + box.size[1]
        ]

        # Check that the level of the corners is the maximum of all points in the bottom face
        if not np.array_equal(lev, np.amax(bottom_face_lev)):
            return 0

        # Count how many of the points in the bottom face are supported at height equal to lev
        count_level = np.count_nonzero(bottom_face_lev == lev)
        # Check the percentage of box bottom area that is supported (at the height equal to lev)
        support_perc = int((count_level / (box.size[0] * box.size[1])) * 100)
        if support_perc < check_area:
            return 0

        dummy_box = deepcopy(box)
        dummy_box.position = [*new_pos, lev]

        # Check that the box fits in the container in the new location
        dummy_box_min_max = [
            dummy_box.position[0],
            dummy_box.position[1],
            dummy_box.position[2],
            dummy_box.position[0] + dummy_box.size[0],
            dummy_box.position[1] + dummy_box.size[1],
            dummy_box.position[2] + dummy_box.size[2],
        ]

        container_min_max = [
            self.position[0],
            self.position[1],
            self.position[2],
            self.position[0] + self.size[0],
            self.position[1] + self.size[1],
            self.position[2] + self.size[2],
        ]

        if not cuboid_fits(container_min_max, dummy_box_min_max):
            return 0

        # Check that the box does not overlap with other boxes in the container
        for other_box in self.boxes:
            if other_box.id_ == dummy_box.id_:
                continue
            other_box_min_max = [
                other_box.position[0],
                other_box.position[1],
                other_box.position[2],
                other_box.position[0] + other_box.size[0],
                other_box.position[1] + other_box.size[1],
                other_box.position[2] + other_box.size[2],
            ]

            if cuboids_intersection(dummy_box_min_max, other_box_min_max):
                return 0

        # if all conditions are met, the position is valid
        return 1

    def action_mask(
        self, box: Box, check_area: int = 100
    ) -> NDArray[Shape["*, *"], Int]:
        """Returns an array with all possible positions for a box in the container
        array[i,j] = 1 if the box can be placed in position (i,j), 0 otherwise

           Parameters
           ----------
           box: Box
               Box to be placed
           check_area: int, default = 100
                Percentage of area of the bottom of the box that must be supported in the new position

           Returns
           -------
           np.array(np.int8)
        """

        action_mask = np.zeros(shape=[self.size[0], self.size[1]], dtype=np.int8)
        # Generate all possible positions for the box in the container
        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                if (
                    self.check_valid_box_placement(
                        box, np.array([i, j], dtype=np.int32), check_area
                    )
                    == 1
                ):
                    action_mask[i, j] = 1
        return action_mask

    def place_box(self, box: Box, new_position: List[int], check_area=100) -> None:
        """Places a box in the container
        Parameters
        ----------
        box: Box
            Box to be placed
        new_position: List[int]
            Coordinates of new position
        check_area:

        """
        assert (
            self.check_valid_box_placement(box, new_position, check_area) == 1
        ), "Invalid position for box"
        # Check height_map to find the height at which the box will be placed
        height = self.height_map[new_position[0], new_position[1]]
        # Update the box position
        box.position = np.asarray([*new_position, height], dtype=np.int32)
        # Add the box to the container
        self.boxes.append(box)
        # Update the height_map
        self._update_height_map(box)

    def plot(self, figure: Type[go.Figure] = None) -> Type[go.Figure]:
        """Adds the plot of a container with its boxes to a given figure

        Parameters
        ----------
        figure: go.Figure, default = None
            A plotly figure where the box should be plotted
        Returns
        -------
            go.Figure
        """
        if figure is None:
            figure = go.Figure()

        # Generate all vertices and edge pairs, the numbering is explained in the function utils.generate_vertices
        vertices = generate_vertices(self.size, self.position).T
        x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
        edge_pairs = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

        # Add a line between each pair of edges to the figure
        for (m, n) in edge_pairs:
            vert_x = np.array([x[m], x[n]])
            vert_y = np.array([y[m], y[n]])
            vert_z = np.array([z[m], z[n]])
            figure.add_trace(
                go.Scatter3d(
                    x=vert_x,
                    y=vert_y,
                    z=vert_z,
                    mode="lines",
                    line=dict(color="yellow", width=3),
                )
            )

        color_list = px.colors.qualitative.Dark24

        for item in self.boxes:
            # item_color = color_list[-2]
            item_color = color_list[(item.volume + item.id_) % len(color_list)]
            figure = item.plot(item_color, figure)

        # Choose the visualization angle
        # camera = dict(eye=dict(x=2, y=2, z=0.1))

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        )

        # Update figure properties for improved visualization
        figure.update_layout(
            showlegend=False,
            scene_camera=camera,
            width=1200,
            height=1200,
            template="plotly_dark",
        )

        max_x = self.position[0] + self.size[0]
        max_y = self.position[1] + self.size[1]
        max_z = self.position[2] + self.size[2]
        figure.update_layout(
            scene=dict(
                xaxis=dict(nticks=int(max_x + 2), range=[0, max_x + 5]),
                yaxis=dict(nticks=int(max_y + 2), range=[0, max_y + 5]),
                zaxis=dict(nticks=int(max_z + 2), range=[0, max_z + 5]),
                aspectmode="cube",
            ),
            width=1200,
            margin=dict(r=20, l=10, b=10, t=10),
        )

        figure.update_scenes(
            xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False
        )
        figure.update_scenes(
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            zaxis_showticklabels=False,
        )

        return figure

    def first_fit_decreasing(self, boxes: List[Box], check_area: int = 100) -> None:
        """Places all boxes in the container using the first fit decreasing heuristic method
        Parameters
        ----------
        boxes: List[Box]
            List of boxes to be placed
        check_area: int, default = 100
            Percentage of area of the bottom of the box that must be supported in the new position
        """
        # Sort the boxes in the decreasing order of their volume
        boxes.sort(key=lambda x: x.volume, reverse=True)

        for box in boxes:
            # Find the positions where the box can be placed
            action_mask = self.action_mask(box, check_area)

            # top lev is the maximum level where the box can be placed
            # according to its height
            top_lev = self.size[2] - box.size[2]
            # max_occupied is the maximum height occupied by a box in the container
            max_occupied = np.max(self.height_map)
            lev = min(top_lev, max_occupied)

            # We find the first position where the box can be placed starting from
            # the top level and going down
            k = lev
            while k >= 0:
                locations = np.zeros(shape=(self.size[0], self.size[1]), dtype=np.int32)
                kth_level = np.logical_and(
                    self.height_map == k, np.equal(action_mask, 1)
                )
                if kth_level.any():
                    locations[kth_level] = 1
                    # Find the first position where the box can be placed
                    position = [
                        np.nonzero(locations == 1)[0][0],
                        np.nonzero(locations == 1)[1][0],
                    ]
                    # Place the box in the first position found
                    self.place_box(box, position, check_area)
                    break
                k -= 1


if __name__ == "__main__":
    len_bin_edges = [10, 10, 10]
    # The boxes generated will fit exactly in a container of size [10,10,10]
    boxes_sizes = boxes_generator(len_bin_edges, num_items=64, seed=42)
    boxes = [
        Box(size, position=[-1, -1, -1], id_=i) for i, size in enumerate(boxes_sizes)
    ]
    # We pack the boxes in a bigger container since the heuristic rule is not optimal
    container = Container(np.array([12, 12, 12], dtype=np.int32))
    # The parameter 'check_area' gives the percentage of the bottom area of the box that must be supported
    container.first_fit_decreasing(boxes, check_area=100)
    # show plot
    fig = container.plot()
    fig.show()
