class TemporalExtent:
    def __init__(self, t_start: int, t_end: int):
        self._t_start = t_start
        self._t_end = t_end
        self._length = t_end - t_start + 1

    def __and__(self, other):
        inter_ext = max(self._t_start, other._t_start), min(self._t_end, other._t_end)
        return TemporalExtent(*inter_ext)

    def __or__(self, other):
        union_ext = min(self._t_start, other._t_start), max(self._t_end, other._t_end)
        return TemporalExtent(*union_ext)

    def intersection(self, other) -> int | float:
        return (self & other).length

    def union(self, other) -> int | float:
        return (self | other).length

    def __contains__(self, fno: int):
        return self._t_start <= fno <= self._t_end

    def __iter__(self):
        return iter(range(self._t_start, self._t_end + 1))

    @property
    def start(self):
        return self._t_start

    @property
    def end(self):
        return self._t_end

    @property
    def extent(self):
        return self._t_start, self._t_end

    @property
    def length(self):
        return self._length

    def __repr__(self):
        return f"TemporalExtent[{self._t_start}, {self._t_end}]"


class BBox:
    def __init__(self, fno, x1, y1, x2, y2):
        self.fno = fno
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1) + 1e-6  # eps for data noise handling

    def __repr__(self):
        return "BBox[fno = {}, x1 = {}, y1 = {}, x2 = {}, y2 = {}]".format(
            self.fno, self.x1, self.y1, self.x2, self.y2
        )

    def intersection(self, other) -> int | float:
        if other is None:
            return 0
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x1 > x2 or y1 > y2:
            return 0
        return (x2 - x1) * (y2 - y1)

    def union(self, other) -> int | float:
        if other is None:
            return self.area
        inter = self.intersection(other)
        return self.area + other.area - inter

    def to_json(self):
        return {
            "fno": int(self.fno),
            "x1": int(self.x1),
            "x2": int(self.x2),
            "y1": int(self.y1),
            "y2": int(self.y2),
        }

    @staticmethod
    def from_json(data):
        if all(key in data for key in ["fno", "x1", "y1", "x2", "y2"]):
            fno, x1, y1, x2, y2 \
                = data["fno"], data["x1"], data["y1"], data["x2"], data["y2"]
        elif all(key in data for key in ["fno", 'x', 'y', 'w', 'h']):
            fno, x1, y1, x2, y2 \
                = data["fno"], data["x"], data["y"], data["x"] + data["w"], data["y"] + data["h"]
        else:
            raise ValueError(f"====> BBox: Invalid JSON data")
        return BBox(fno, x1, y1, x2, y2)


class ResponseTrack:
    def __init__(self, bboxes: list[BBox], score: float = None):
        # A set of bounding boxes with time, and an optional confidence score
        self._bboxes: list = sorted(bboxes, key=lambda x: x.fno)
        if self._check_empty(self._bboxes):
            self._empty_init()
        else:
            self._non_empty_init(self._bboxes)
        self._score = score
        self._check_contiguous()
        self._fnos = {bbox.fno for bbox in self._bboxes}

    @property
    def temporal_extent(self):
        return TemporalExtent(self._t_start, self._t_end)

    @property
    def bboxes(self):
        return self._bboxes

    @property
    def length(self):
        return self._length

    @property
    def score(self):
        return self._score

    @property
    def volume(self):
        v = 0.0
        for bbox in self._bboxes:
            v += bbox.area
        return v

    def has_score(self):
        return self._score is not None

    def _check_empty(self, bboxes):
        return len(bboxes) == 0

    def _empty_init(self):
        self._t_start = 0
        self._t_end = -1
        self._length = 0
        # print("Encountered empty track")

    def _non_empty_init(self, bboxes):
        self._t_start = bboxes[0].fno
        self._t_end = bboxes[-1].fno
        self._length = len(bboxes)

    def _check_contiguous(self):
        if self._length != (self._t_end - self._t_start + 1):
            raise ValueError(f"====> ResponseTrack: BBoxes not contiguous")

    def __iter__(self):
        return iter(self._bboxes)

    def __getitem__(self, fno: int):
        return self._bboxes[fno - self._t_start]

    def __contains__(self, fno: int):
        return fno in self._fnos

    def __repr__(self):
        return (
            "ResponseTrack[\n"
            + "\n".join([bbox.__repr__() for bbox in self._bboxes])
            + "]"
        )

    def intersection(self, other) -> int | float:
        tinter_ext = self.temporal_extent & other.temporal_extent
        stinter = 0
        for fno in tinter_ext:
            b1, b2 = self[fno], other[fno]
            stinter += b1.intersection(b2)
        return stinter

    def union(self, other) -> int | float:
        tunion_ext = self.temporal_extent | other.temporal_extent
        area = 0
        for fno in tunion_ext:
            if fno in self:
                area += self[fno].area
            if fno in other:
                area += other[fno].area
        stinter = self.intersection(other)
        return area - stinter

    def get(self, fno, default = None):
        if fno not in self._fnos:
            return default
        return self._bboxes[fno - self._t_start]

    def to_json(self):
        score = self._score
        if score is not None:
            score = float(score)
        return {
            "bboxes": [bbox.to_json() for bbox in self._bboxes],
            "score": score,
        }

    @staticmethod
    def from_json(data: dict):
        return ResponseTrack(
            [BBox.from_json(bbox) for bbox in data.get("bboxes") or data['response_track']],
            data.get("score", None),
        )
