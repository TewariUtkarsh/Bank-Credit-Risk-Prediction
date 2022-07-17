from datetime import datetime


def get_current_time_stamp() -> str:
    """
    This function is responsible for returning the 
    current timestamp value in string format.

    Returns
    ----------
    str : str
        Current timestamp in string format.
    """
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"



