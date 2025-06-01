class GeometryError(Exception):
    # Parent of every Exception here
    pass

class ConstructError(GeometryError):
    # Occurs when object created or edited incorrectly
    pass

class SceneError(GeometryError):
    # Occurs when scene created or edited incorrectly
    pass

class IntersectionError(GeometryError):
    # Error in decorator for all object.intersects here
    pass