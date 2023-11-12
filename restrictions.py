import networkx as nx

G = nx.Graph()

G.add_node("START", pos=(0, 0))
G.add_node("TOP", pos=(1, 0))
G.add_node("BOTTOM", pos=(2, 0))
G.add_node("FOOTWEAR", pos=(3, 0))
G.add_node("ACCESSORIES", pos=(4, 0))
G.add_node("COMPLETE_SET", pos=(5, 0))
G.add_node("END", pos=(6, 0))
G.add_node("ERROR", pos=(7, 0))

# START = ( top -> B | completeSet -> F)
G.add_edge("START", "TOP", weight=0)
G.add_edge("START", "COMPLETE_SET", weight=0)

# B = (bottom -> C)
G.add_edge("TOP", "BOTTOM", weight=0)

# C = (footwear -> D)
G.add_edge("BOTTOM", "FOOTWEAR", weight=0)

# D = (accessories -> E) | (end -> END)
G.add_edge("FOOTWEAR", "ACCESSORIES", weight=0)
G.add_edge("FOOTWEAR", "END", weight=0)

# E = (end -> END)
G.add_edge("ACCESSORIES", "END", weight=0)

# F = (footwear -> D)
G.add_edge("COMPLETE_SET", "FOOTWEAR", weight=0)

# connect each state to the ERROR state
G.add_edge("START", "ERROR", weight=-100)  # it menas not possible classes
G.add_edge("TOP", "ERROR", weight=-1)
G.add_edge("BOTTOM", "ERROR", weight=-0.66)
G.add_edge("FOOTWEAR", "ERROR", weight=-0.33)
G.add_edge("ACCESSORIES", "ERROR", weight=-100)
G.add_edge("COMPLETE_SET", "ERROR", weight=-1)


def error_function(G=G, state="START", actions_list=[], error_value=0):
    if state == "END" and len(actions_list) == 0:
        if error_value > 0:
            return error_value
        else:
            return 0

    if state == "ERROR":
        return error_value

    action = actions_list.pop(0)

    if action in G[state] and action != "ERROR" and G[state][action]:
        return error_function(G, action, actions_list, error_value)
    elif action in G[state] and action != "ERROR" and not G[state][action]:
        return error_function(
            G, action, actions_list, error_value + G[state]["ERROR"]["weight"]
        )
    else:
        return error_function(G, "ERROR", [], error_value + G[state]["ERROR"]["weight"])


class Outfit:
    def __init__(self, items):
        self.items = items

    def get_restrictions(self):
        actions_list = ["START"]
        for item in self.items:
            if item == "TOP":
                actions_list.append("TOP")
            elif item == "BOTTOM":
                actions_list.append("BOTTOM")
            elif item == "FOOTWEAR":
                actions_list.append("FOOTWEAR")
            elif item == "ACCESSORIES":
                actions_list.append("ACCESSORIES")
            elif item == "COMPLETE_SET":
                actions_list.append("COMPLETE_SET")
        actions_list.append("END")
        return error_function(G, "START", actions_list, 0)

    def __init__(self, rows=None):
        """
        Initializes an Outfit object with the given rows.

        Args:
        - rows: a list of rows representing the outfit components.
        """
        if rows is None:
            rows = []
        self.top = False
        self.bottom = False
        self.accessories = False
        self.complete_set = False
        self.footwear = False
        for row in rows:
            if row["des_product_family"] == "Footwear":
                self.footwear = True
            else:
                if row["des_product_category"] == "Top":
                    self.top = True
                if row["des_product_category"] == "Bottom":
                    self.bottom = True
                if row["des_product_category"] == "Accesories, Swim and Intimate":
                    self.accessories = True
                if row["des_product_category"] == "Dresses, jumpsuits and Complete set":
                    self.complete_set = True

    def get_error(self):
        """
        Returns the error value of the outfit.

        Returns:
        - The error value of the outfit.
        """
        actions_list = self.get_actions_list()
        return error_function(actions_list)

    def get_actions_list(self):
        """
        Returns a list of actions based on the outfit components.

        Returns:
        - A list of actions based on the outfit components.
        """
        actions_list = []
        if self.top:
            actions_list.append("TOP")
        if self.bottom:
            actions_list.append("BOTTOM")
        if self.complete_set:
            actions_list.append("COMPLETE SET")
        if self.footwear:
            actions_list.append("FOOTWEAR")
        if self.accessories:
            actions_list.append("ACCESSORIES")

        return actions_list
