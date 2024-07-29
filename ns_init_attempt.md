------------------------------------------------------------------------------------------------------------
6843_2: Go down the hall on the left and into the living room. Go through the door on the right until you reach the treadmill.
view_select(context='hall on the left', full360=True)  # selects the appropriate camera view with context
follow(route='hall', stop='living room')  # follows the route until you reach the stop or route ends, i.e., 
					    # all viewpoints in the route with final viewpoint also being on stop
go_through(exit='door on the right')  # reaches the viewpoint exit and goes through it, i.e., last viewpoint does not have exit visible
go_to(stop='treadmill')  # goes to the stop viewpoint, last viewpoint within some distance of the stop

7121_0: Turn around and go up the stairs. Stop at the top of the stairs.
view_select(context='stairs behind', full360=True)
follow(route='stairs', stop='')
------------------------------------------------------------------------------------------------------------


VLM(image, text) -> text
LLM(text) -> text
DEPTH(image) -> np.array
SEG(image, text) -> np.array


def view_select(context: str, full360: bool) -> None:
    vlm = VLM(prompt="Describe the image in detail.")
    desc = []
    views = get_views(full360=full360)
    for im in views:
        desc.append(vlm(im))
    llm = LLM()
    ans = llm(prompt="Which view has the " + context + "?", options=desc, default=0).int()
    make_selection(ans)

def go_straight(image: np.array) -> None:
    depth = DEPTH(image)
    viewpoints = get_viewpoints()
    v = closest_to_center(viewpoints, depth)
    move_to(v)
    
def follow(route: str, stop: str) -> None:
    vlm = VLM(prompt="Is there " + route + " straight down?")
    if stop:
        vlm2 = VLM(prompt="Do you see a " + stop + " at the front?")
    while True:
        views = get_views(full360=False)
        view_select(context=route, full360=False)
        if stop and vlm2(views[0]).bool() and not vlm(views[0]).bool():
            break
        if not stop and not vlm(views[0]).bool():
            break
        go_straight(views[0])
    
def go_through(exit: str) -> None:
    go_to(stop=exit)
    views = get_views(full360=False)
    view_select(context=exit, full360=False)
    seg = SEG()
    obj = seg(views[0], exit)
    if obj is None:
        return
    go_straight(views[0])
    
def go_to(stop: str) -> None:
    seg = SEG()
    depth = DEPTH()
    while True:
        views = get_views(full360=False)
        view_select(context=stop, full360=False)
        obj = seg(views[0], stop)
        if obj is None:
            break
        d = depth(views[0])
        viewpoints = get_viewpoints()
        v = closest_to_obj(viewpoints, obj, d)
        if v == 0:  # self
            break
        move_to(v)
