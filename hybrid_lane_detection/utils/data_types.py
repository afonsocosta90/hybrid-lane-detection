from dataclasses import dataclass, field
import numpy as np
import time

@dataclass(slots=True)
class LaneData:
    """
        This class acts as a bridge between the faster Classical path and
        the fallback DL path. This ensures that the the downstream logic
        receives consistent data format, regardless of which algorithm
        generated the detection.
    """

    left_fit: np.ndarray               # Represents the left lane curvature as a 2nd order polynomial x = ay^2 + by + c
    right_fit: np.ndarray              # Represents the right lane curvature as a 2nd order polynomial x = ay^2 + by + c
    confidence: float                  # Float [0, 1], represents the confidence level. Used as the decision variable in the confidence gate
    source: str                        # Identifying tag ("classical" or "dl"). Used for debugging and logging which backend is being used
    # Per-instance timestamp; field(default_factory=...) avoids the shared-default bug
    timestamp: float = field(default_factory=time.time)
