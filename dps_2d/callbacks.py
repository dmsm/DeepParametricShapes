import ttools.callbacks as cb


class InputImageCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'input'

    def visualized_image(self, batch, fwd_result):
        return batch['im']


class RenderingCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'pred'

    def visualized_image(self, batch, fwd_result):
        return fwd_result['occupancy_fields'].unsqueeze(1)

class OverlapCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'overlap'

    def visualized_image(self, batch, fwd_result):
        return fwd_result['overlap_fields'].unsqueeze(1)
