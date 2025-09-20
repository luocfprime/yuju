from contextlib import contextmanager

import polyscope.imgui as psim


@contextmanager
def ui_item_width(width):
    psim.PushItemWidth(width)
    try:
        yield
    finally:
        psim.PopItemWidth()


@contextmanager
def ui_tree_node(label, open_first_time=True):
    """
    A context manager for creating a collapsible ImGui tree node.

    This correctly handles the ImGui pattern of checking the return of TreeNode
    and calling TreePop conditionally, while ensuring the context manager
    protocol is always followed.
    """
    psim.SetNextItemOpen(open_first_time, psim.ImGuiCond_FirstUseEver)
    expanded = psim.TreeNode(label)

    try:
        yield expanded
    finally:
        if expanded:
            psim.TreePop()


@contextmanager
def ui_combo(label, current_value):
    """
    A context manager for creating an ImGui combo box.
    """
    expanded = psim.BeginCombo(label, current_value)

    try:
        yield expanded
    finally:
        if expanded:
            psim.EndCombo()
