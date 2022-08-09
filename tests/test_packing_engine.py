import pytest
import numpy as np
from src.packing_engine import Box, Container
import random as rd

rd.seed(42)
num_rd_tests = 5
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
pos_x = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_y = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_z = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)*0

length_edges = zip(height, length, width)
positions = zip(pos_x, pos_y, pos_z)
test_data1 = zip(length_edges, positions)
test_data2 = zip(length_edges, positions)


# # Test of initialization of Container class
@pytest.mark.parametrize("len_edges, position", test_data1)
def test_container_initialization_random_data(len_edges, position):
    container = Container(len_edges, position, 0)
    assert np.array_equal(container.len_edges, len_edges)
    assert np.array_equal(container.position, position)
    assert container.id_ == 0
    assert container.height_map.shape == (len_edges[0], len_edges[1])


num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
pos_x = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_y = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_z = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)

length_edges = zip(height, length, width)
positions = zip(pos_x, pos_y, pos_z)
test_data2 = zip(length_edges, positions)


# Test of initialization of Box class
@pytest.mark.parametrize("len_edges, position", test_data2)
def test_box_initialization_random_data(len_edges, position):
    box = Box(len_edges, position, 0)
    assert np.array_equal(box.len_edges, len_edges)
    assert np.array_equal(box.position, position)


# Test of update_height_map
container_len_edges = [[6, 6, 10], [6, 8, 10]]
box_len_edges = [[2, 2, 5], [2, 8, 3]]
box_pos = [[3, 3, 0], [2, 0, 0]]
hm1 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 0],
                [0, 0, 0, 5, 5, 0], [0, 0, 0, 0, 0, 0]], dtype=np.int32)
hm2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
height_map = [hm1, hm2]
test_data3 = zip(container_len_edges, box_len_edges, box_pos, height_map)


@pytest.mark.parametrize("container_len_edges, box_len_edges, box_pos, height_map", test_data3)
def test_update_height_map(container_len_edges, box_len_edges, box_pos, height_map):
    container = Container(container_len_edges)
    box = Box(box_len_edges, box_pos, 0)
    container._update_height_map(box)
    assert np.array_equal(container.height_map, height_map)


# Test of Box property area_bottom
box_len_edges = [[2, 2, 5], [2, 8, 3]]
box_pos = [[3, 3, 0], [2, 0, 0]]
area = [4, 16]
test_data4 = zip(box_len_edges, box_pos, area)


@pytest.mark.parametrize("box_len_edges, box_pos, area", test_data4)
def test_box_area_bottom(box_len_edges, box_pos, area):
    box = Box(box_len_edges, box_pos, 0)
    assert box.area_bottom == area


# Test of Box property vertices
box_len_edges = [[2, 2, 5], [2, 8, 3]]
box_pos = [[3, 3, 0], [2, 0, 0]]
vert_1 = np.asarray([[3, 3, 0], [5, 3, 0], [3, 5, 0], [5, 5, 0], [3, 3, 5], [5, 3, 5], [3, 5, 5], [5, 5, 5]])
vert_2 = np.asarray([[2, 0, 0], [4, 0, 0], [2, 8, 0], [4, 8, 0], [2, 0, 3], [4, 0, 3], [2, 8, 3], [4, 8, 3]])
vertices = [vert_1, vert_2]
test_data5 = zip(box_len_edges, box_pos, vertices)


@pytest.mark.parametrize("box_len_edges, box_pos, vertices", test_data5)
def test_box_vertices(box_len_edges, box_pos, vertices):
    box = Box(box_len_edges, box_pos, 0)
    assert np.array_equal(box.vertices, vertices)


# Test of Container.check_valid_box_placement -- box in empty container
container_len_edges = [[10, 10, 10], [10, 10, 10]]
box_len_edges = [[2, 4, 3], [2, 4, 3]]
box_pos = [[5, 5, 0], [2, 8, 0]]
valid = [1, 0]
test_data6 = zip(container_len_edges, box_len_edges, box_pos, valid)


@pytest.mark.parametrize("container_len_edges, box_len_edges, box_pos, valid", test_data6)
def test_check_valid_box_placement(container_len_edges, box_len_edges, box_pos, valid):
    container = Container(container_len_edges)
    box = Box(box_len_edges, [-1, -1, -1], 0)
    is_valid1 = container.check_valid_box_placement(box, box_pos[0:2])
    np.testing.assert_equal(is_valid1, valid)


# Test of Container.check_valid_box_placement -- large box in empty container
container_len_edges = [[10, 10, 10], [10, 10, 10]]
box_len_edges = [[2, 10, 3], [10, 2, 3]]
box_pos = [[5, 0, 0], [2, 0, 0]]
valid = [1, 0]
test_data7 = zip(container_len_edges, box_len_edges, box_pos, valid)


@pytest.mark.parametrize("container_len_edges, box_len_edges, box_pos, valid", test_data7)
def test_check_valid_box_placement(container_len_edges, box_len_edges, box_pos, valid):
    container = Container(container_len_edges)
    box = Box(box_len_edges, [-1, -1, -1], 0)
    is_valid2 = container.check_valid_box_placement(box, box_pos[0:2])
    np.testing.assert_equal(is_valid2, valid)


# Test to pack a sequence of boxes into a container
def test_pack_boxes():
    box0 = Box([3, 3, 3], [0, 0, 0], 0)
    box1 = Box([3, 2, 3], [0, 0, 3], 1)
    box2 = Box([3, 4, 2], [3, 0, 0], 2)
    box3 = Box([3, 2, 4], [0, 3, 0], 3)
    box4 = Box([3, 2, 3], [0, 5, 0], 4)

    container = Container([10, 10, 10])
    container.place_box(box0, [0, 0])
    # Check that box0 is in the container in the right place
    np.testing.assert_array_equal(container.boxes[0].position, np.asarray([0, 0, 0]))

    # Check that the height map is correct
    hm = np.zeros((10, 10), dtype=np.int32)
    hm[0:3, 0:3] = 3
    np.testing.assert_array_equal(container.get_height_map(), hm)

    # Check that box1 can be placed in the container
    a = container.check_valid_box_placement(box1, [0, 0], 100)
    np.testing.assert_equal(a, 1)
    # Check that box1 is in the container in the right place
    container.place_box(box1, [0, 0])
    np.testing.assert_array_equal(container.boxes[1].position, np.asarray([0, 0, 3]))

    # Check that box2 can be placed in the container
    np.testing.assert_equal(container.check_valid_box_placement(box2, [3, 0], 100), 1)
    # Check that box2 is in the container in the right place
    container.place_box(box2, [3, 0])
    np.testing.assert_array_equal(container.boxes[2].position, np.asarray([3, 0, 0]))

    # Check that box3 can be placed in the container
    np.testing.assert_equal(container.check_valid_box_placement(box3, [0, 3], 100), 1)
    # Check that box3 is in the container in the right place
    container.place_box(box3, [0, 3])
    np.testing.assert_array_equal(container.boxes[3].position, np.asarray([0, 3, 0]))

    # Check that box4 can be placed in the container
    np.testing.assert_equal(container.check_valid_box_placement(box4, [0, 5], 100), 1)
    # Check that box4 is in the container in the right place
    container.place_box(box4, [0, 5])
    np.testing.assert_array_equal(container.boxes[4].position, np.asarray([0, 5, 0]))

# box1 = Box([3, 2, 3], [-1, -1, -1], 1)
# box2 = Box([3, 4, 2], [-1, -1, -1], 2)
# box3 = Box([3, 2, 4], [-1, -1, -1], 3)
# box4 = Box([3, 2, 3], [-1, -1, -1], 4)


# Test of method Container.all_possible_positions with a sequence of boxes
def test_all_possible_positions():
    container = Container([10, 10, 10])

    # Create box0 and action mask for box0
    box0 = Box([3, 3, 3], [-1, -1, -1], 0)
    box0_action_mask = np.zeros(shape=[10, 10], dtype=np.int32)
    box0_action_mask[0:8, 0:8] = 1
    # check all possible positions for box0
    np.testing.assert_array_equal(container.all_possible_positions(box0, 100), box0_action_mask)
    # place box0 at [0,0]
    container.place_box(box0, [0, 0])
    # check height map after placing box0
    hm0 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm0[0:3, 0:3] = 3
    np.testing.assert_array_equal(container.get_height_map(), hm0)

    # Create box1 and action mask for box1
    box1 = Box([3, 2, 3], [-1, -1, -1], 1)
    box1_action_mask = np.ones(shape=[10, 10], dtype=np.int32)
    box1_action_mask[1, 0:3] = 0
    box1_action_mask[2, 0:3] = 0
    box1_action_mask[0, 2] = 0
    box1_action_mask[8, :] = 0
    box1_action_mask[9, :] = 0
    box1_action_mask[:, 9] = 0
    # check all possible positions for box1
    np.testing.assert_array_equal(container.all_possible_positions(box1, 100), box1_action_mask)
    # place box1 at [0,0]
    container.place_box(box1, [0, 0])
    # check height map after placing box1
    hm1 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm1[0:3, 0:2] = 6
    hm1[0:3, 2] = 3
    np.testing.assert_array_equal(container.get_height_map(), hm1)

    # Create box2 and action_mask for box2
    box2 = Box([3, 4, 2], [-1, -1, -1], 2)
    box2_action_mask = np.zeros(shape=[10, 10], dtype=np.int32)
    box2_action_mask[0:8, 0:7] = 1
    box2_action_mask[0:3, 0:3] = 0
    # check all possible positions for box2
    np.testing.assert_array_equal(container.all_possible_positions(box2, 100), box2_action_mask)
    # place box2 at [3,0]
    container.place_box(box2, [3, 0])
    # check height map after placing box2
    hm2 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm2[0:3, 0:2] = 6
    hm2[0:3, 2] = 3
    hm2[3:6, 0:4] = 2
    np.testing.assert_array_equal(container.get_height_map(), hm2)

    # Create box3 and action_mask for box3
    box3 = Box([3, 2, 4], [-1, -1, -1], 3)
    box3_action_mask = np.zeros(shape=[10, 10], dtype=np.int32)
    box3_action_mask[0:8, 0:9] = 1
    box3_action_mask[0:6, 0:4] = 0
    box3_action_mask[0, 0] = 1
    box3_action_mask[0, 3] = 1
    box3_action_mask[3, 0:3] = 1
    # check all possible positions for box3
    np.testing.assert_array_equal(container.all_possible_positions(box3, 100), box3_action_mask)
    # place box3 at [0,3]
    container.place_box(box3, [0, 3])
    # check height map after placing box3
    hm3 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm3[0:3, 0:2] = 6
    hm3[0:3, 2] = 3
    hm3[3:6, 0:4] = 2
    hm3[0:3, 3:5] = 4
    np.testing.assert_array_equal(container.get_height_map(), hm3)

    # Create box4 and action_mask for box4
    box4 = Box([3, 2, 3], [-1, -1, -1], 4)
    box4_action_mask = np.zeros(shape=[10, 10], dtype=np.int32)
    box4_action_mask[0:8, 0:9] = 1
    box4_action_mask[0:6, 0:4] = 0
    box4_action_mask[0, 0] = 1
    box4_action_mask[0, 3] = 1
    box4_action_mask[3, 0:3] = 1
    box4_action_mask[0:3, 4] = 0
    # check all possible positions for box4
    np.testing.assert_array_equal(container.all_possible_positions(box4, 100), box4_action_mask)
    # place box4 at [0,5]
    container.place_box(box4, [0, 5])
    # check height map after placing box4
    hm4 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm4[0:3, 0:2] = 6
    hm4[0:3, 2] = 3
    hm4[3:6, 0:4] = 2
    hm4[0:3, 3:5] = 4
    hm4[0:3, 5:7] = 3
    np.testing.assert_array_equal(container.get_height_map(), hm4)




















