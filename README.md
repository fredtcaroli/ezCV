# ezCV

> **_NOTE:_** This is really not finished yet. 
> Feel free to contact me with suggestions :)

ezCV is an easy way of building computer vision pipelines.

This is the backend library for ezCV-GUI, a GUI for building CV pipelines.  

### Installing

You'll need OpenCV installed with python bindings. We recommend using conda:

```bash
conda create -n my_env
conda activate my_env
conda install opencv
```

You'll also need poetry installed:

```bash
pip install poetry
```

You can then download the source code and run

```bash
poetry install
```

We'll probably provide an easier way to install this later.

### Documentation

This is just a brief documentation of what I've done so far.


##### Configs

Here's what an ezCV config file looks like:

```yaml
# config.yml

version: '0.0'

pipeline:
  - name: bgr2gray
    config:
      implementation: ezcv.operator.implementations.color_space.ColorSpaceChange
      params:
        src: BGR
        target: GRAY
  - name: blur
    config:
      implementation: ezcv.operator.implementations.blur.GaussianBlur
      params:
        kernel_size: 5
        sigma: 1
```

This pipeline takes an image, transforms it from BGR to grayscale and blurs it.

Config files are created using [ezCV-GUI](https://github.com/fredtcaroli/ezCV-GUI/), but there shouldn't 
be any reasons why not to write or edit them manually.

##### Executing Pipelines

Let's say the config file above is in the current dir and is called `config.yml`. You can then
parse it and use it easily:

```python
import cv2
from ezcv import CompVizPipeline

with open('config.yml') as f:
    pipeline = CompVizPipeline.load(f)

img = cv2.imread('example_img.png')
output, ctx = pipeline.run(img)
```

Now `output` is the processed image, while `ctx` gives us info that operators
registered along the way. More on that shortly.

##### Creating Operators

Creating an operator should be easy. Let's implement one.

The goal of this operator is to threshold a grayscale image, given a threshold value and method:

```python
import cv2
from ezcv.operator import register_operator, Operator, EnumParameter, IntegerParameter
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image

# The @register_operator makes sure your operator will be available for choosing in ezCV-GUI
@register_operator
class ThresholdOperator(Operator):
    """ Applies a Threshold operation

    Parameters:
        - threshold: Threshold value
        - method: Thresholding method to use
    """
    # This flag tells ezCV that this operator only accepts grayscale images
    only_gray = True
    
    # You declare your parameters with Parameter classes, but you can use them like they
    # are regular builtin types, like strings and integers
    threshold = IntegerParameter(default_value=127, lower=0, upper=255)
    method = EnumParameter(possible_values=["binary", "otsu"], default_value="binary")

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        flags = cv2.THRESH_BINARY
        if self.method == 'otsu':
            flags += cv2.THRESH_OTSU
        _, output = cv2.threshold(img, self.threshold, 255, flags)
        return output
```

Now you can use your operator by writing a config file just like the one showed before:

```yaml
#custom_operator_config.yml

version: '0.0'

pipeline:
  - name: bgr2gray
    config:
      implementation: ezcv.operator.implementations.color_space.ColorSpaceChange
      params:
        src: BGR
        target: GRAY
  - name: threshold
    config:
      implementation: path.to.operator.ThresholdOperator
      params:
        threshold: 127
        method: binary
```

##### More Complex Use Cases

We have one more example to go. In this example we'll show how to:

1. Set up a custom type of parameter;
2. Add information to the pipeline context.

For this example we'll use a fictional, somewhat useless, operator.

This operator is going to use a coordinates parameter `coord`, which is tuple of 2 value, x and y. 
This parameter will be saved to the final config file as an object with `x` and `y` properties.
A 2-tuple is perfectly serializable the way it is, but we're doing this to show that you have full control
of how parameters are saved, and that you can save complex parameter types if you just provide a yaml-serializable
object for it.

Given a coordinate `coord`, the operator is going to register in the pipeline context what was the color of the pixel
on that coordinate. We'll later show how to read that value outside the pipeline.

So here's the operator implementation:

```python
from typing import Tuple, Dict

from ezcv.operator import register_operator, Operator, ParameterSpec
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


class CoordinateParameter(ParameterSpec[Tuple[int, int]]):
    def to_config(self, value: Tuple[int, int]) -> Dict[str, int]:
        return {
            'x': value[0],
            'y': value[1]
        }
    def from_config(self, config: Dict[str, int]) -> Tuple[int, int]:
        return config['x'], config['y']
        

@register_operator
class ColorPicker(Operator):
    coord = CoordinateParameter(default_value=(0, 0))

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        x, y = self.coord
        color = img[y, x]
        ctx.add_info('color', color)
        return img
```

So there's a lot going on. Let's cover some points:

* We added a lot of typing. We love typing. Inheriting from `ParameterSpec[Tuple[int, int]]`
will ensure that your parameters will have the correct type inferred when coding on an 
advanced IDE or using mypy.
* Our custom parameter defines two methods, one for encoding your parameter and the other
to decode it.
* Our color picker operator adds the color information to the pipeline context. 

This is how a config that uses our operator looks like:

```yaml
# color_picker.yml

version: '0.0'

pipeline:
  - name: color_picker
    config:
      implementation: path.to.operator.ColorPicker
      params:
        coord:
          x: 30
          y: 30
```

Now let's see how we can run it and check the picked color.

```python
import cv2
from ezcv import CompVizPipeline

with open('color_picker.yml') as f:
    pipeline = CompVizPipeline.load(f)

img = cv2.imread('example_img.png')
output, ctx = pipeline.run(img)
print(ctx.info['color_picker']['color'])  # prints the picked color
```

You can also change the operator's parameters value programmatically:

```python
pipeline.operators['color_picker'].coord = (10, 10)
output, ctx = pipeline.run(img)
print(ctx.info['color_picker']['color'])  # prints a different color

with open('color_picker.yml', 'w') as f:
    pipeline.save(f)  # saves updated parameter
```

And there you have it! There's still some tinkering with pyQt6 if you want a nice graphical interface for
picking the `coord` parameter. There isn't a tutorial on that yet, but you can read how the 
`BooleanParameterWidget` is implemented [here](https://github.com/fredtcaroli/ezCV-GUI/blob/c8c1e39ce7ff61b497878d42f0b3c3f5007c08f8/ezcv_gui/widgets/parameter.py#L251-L264),
and see that it's not so complicated.
