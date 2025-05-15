




def run_tracking(objs, params):
    logger.info(f"====Starting Tracking====\n{pformat(params)}")
    with btrack.BayesianTracker() as tracker:
        tracker.configure(params.config_file)
        if params.features is not None:
            tracker.features = params.features
        tracker.append(objs)
        tracker.max_search_radius = params.max_search_radius
        tracker.max_lost = params.max_lost
        tracker.volume = params.volume
        tracker.track(tracking_updates=params.tracking_updates)
        if params.optimize:
            tracker.optimise()

        Path(params.tracks_out).parent.mkdir(exist_ok=True, parents=True)
        tracker.export(params.tracks_out, obj_type="obj_type_1")
        data, properties, graph = tracker.to_napari()
        with open(params.params_out, "w") as fp:
            json.dump(asdict(params), fp, indent=2)
        tracks = tracker.tracks
    logger.info(f"====== Done ======\n{pformat(params)}")
    return data, properties, graph, tracks, params
