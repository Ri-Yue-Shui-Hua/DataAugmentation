import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import SimpleITK as sitk
import sitkUtils
import time
import numpy as np
import itertools
from transforms.noise_transforms import *
from transforms.abstract_transforms import Compose
from transforms.utility_transforms import *
#
# DataAugmentation
#

class DataAugmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DataAugmentation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DataAugmentation">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # DataAugmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='DataAugmentation',
        sampleName='DataAugmentation1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'DataAugmentation1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='DataAugmentation1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='DataAugmentation1'
    )

    # DataAugmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='DataAugmentation',
        sampleName='DataAugmentation2',
        thumbnailFileName=os.path.join(iconsPath, 'DataAugmentation2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='DataAugmentation2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='DataAugmentation2'
    )


#
# DataAugmentationWidget
#

class DataAugmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.sigma_lineEdit = None
        self.time_cost_lineEdit = None
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DataAugmentation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = DataAugmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.labelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.xTranslationSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.yTranslationSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.zTranslationSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.xRotationSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.yRotationSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.zRotationSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.time_cost_lineEdit = self.ui.time_cost_lineEdit
        self.sigma_lineEdit = self.ui.sigma_lineEdit

        # Buttons
        self.ui.transformPushButton.connect('clicked(bool)', self.onTransformPushButton)
        self.ui.gaussBlurPushButton.connect('clicked(bool)', self.onGaussBlurPushButton)
        self.ui.gaussNoisePushButton.connect('clicked(bool)', self.onGaussNoisePushButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.initializeLayout()
        self.sigma_lineEdit.setText(0.3)

    def initializeLayout(self):
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        viewNode1 = slicer.mrmlScene.GetFirstNodeByName("View1")
        viewNode1.SetBackgroundColor(0, 0, 0)
        viewNode1.SetBackgroundColor2(0, 0, 0)
        viewNode1.SetBoxVisible(False)
        viewNode1.SetAxisLabelsVisible(False)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.labelSelector.setCurrentNode(self._parameterNode.GetNodeReference("LabelVolume"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            pass
        else:
            pass

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("LabelVolume", self.ui.labelSelector.currentNodeID)
        self._parameterNode.SetParameter("xTranslation", str(self.ui.xTranslationSliderWidget.value))
        self._parameterNode.SetParameter("yTranslation", str(self.ui.yTranslationSliderWidget.value))
        self._parameterNode.SetParameter("zTranslation", str(self.ui.zTranslationSliderWidget.value))
        self._parameterNode.SetParameter("xRotation", str(self.ui.xRotationSliderWidget.value))
        self._parameterNode.SetParameter("yRotation", str(self.ui.yRotationSliderWidget.value))
        self._parameterNode.SetParameter("zRotation", str(self.ui.zRotationSliderWidget.value))

        self._parameterNode.EndModify(wasModified)

    def onTransformPushButton(self):
        # 用字典reload后可能会出现滞后性错误
        # 根据平移或旋转信息生成对应的输出图像数据，命名为transformed
        x_shift = float(self.ui.xTranslationSliderWidget.value)
        y_shift = float(self.ui.yTranslationSliderWidget.value)
        z_shift = float(self.ui.zTranslationSliderWidget.value)
        print("x_shift, y_shift, z_shift: ", x_shift, y_shift, z_shift)
        x_angle = float(self.ui.xRotationSliderWidget.value)
        y_angle = float(self.ui.yRotationSliderWidget.value)
        z_angle = float(self.ui.zRotationSliderWidget.value)
        print("x_angle, y_angle, z_angle: ", x_angle, y_angle, z_angle)
        # step1. 先测试平移操作
        volume_node = self._parameterNode.GetNodeReference("InputVolume")
        output_volume_node = slicer.modules.volumes.logic().CloneVolume(volume_node, "translated")
        itk_image = sitkUtils.PullVolumeFromSlicer(output_volume_node)
        label_node = self._parameterNode.GetNodeReference("LabelVolume")
        output_label_node = slicer.modules.volumes.logic().CloneVolume(label_node, "translated_label")
        label_image = sitkUtils.PullVolumeFromSlicer(output_label_node)
        offset = [x_shift, y_shift, z_shift]  # 在x、y和z方向上的平移量
        tic = time.time()
        if x_shift != 0 or y_shift != 0 or z_shift != 0:
            print("in translation")
            transformed_image = sitk_translation_transform(itk_image, offset)
            transformed_label_image = sitk_translation_transform(label_image, offset, is_label=True)
        else:
            print("in rotation")
            default_value = np.float64(sitk.GetArrayViewFromImage(itk_image).min())
            transformed_image = sitk_rotation3d(itk_image, x_angle, y_angle, z_angle, None, default_value)
            transformed_label_image = sitk_rotation3d(label_image, x_angle, y_angle, z_angle, None)
        toc = time.time()
        time_cost = toc - tic
        self.time_cost_lineEdit.setText(str(time_cost) + " s")
        sitkUtils.PushVolumeToSlicer(transformed_image, output_volume_node)
        sitkUtils.PushVolumeToSlicer(transformed_label_image, output_label_node)

    def onGaussBlurPushButton(self):
        print("in gauss blur")
        gauss_blur = GaussianBlurTransform((0.5, 3), different_sigma_per_channel=False, p_per_sample=1.0)
        volume_node = self._parameterNode.GetNodeReference("InputVolume")
        output_volume_node = slicer.modules.volumes.logic().CloneVolume(volume_node, "gauss_blured")
        image_arr = slicer.util.arrayFromVolume(output_volume_node)
        label_node = self._parameterNode.GetNodeReference("LabelVolume")
        if label_node:
            label_arr = slicer.util.arrayFromVolume(label_node)
        else:
            label_arr = np.ones_like(image_arr)
        image_arr = image_arr[np.newaxis, :, :, :]
        label_arr = label_arr[np.newaxis, :, :, :]
        tic = time.time()
        out_dict = gauss_blur(data=image_arr, gt=label_arr)
        toc = time.time()
        time_cost = toc - tic
        self.time_cost_lineEdit.setText(str(time_cost) + " s")
        d, gt = out_dict.get('data'), out_dict.get('gt')
        d = d[0]
        gt = gt[0]
        slicer.util.updateVolumeFromArray(output_volume_node, d)

    def onGaussNoisePushButton(self):
        print("in gauss noise")
        gauss_noise = GaussianNoiseTransform((0.5, 3), different_sigma_per_channel=False, p_per_sample=1.0)
        volume_node = self._parameterNode.GetNodeReference("InputVolume")
        output_volume_node = slicer.modules.volumes.logic().CloneVolume(volume_node, "gauss_blured")
        image_arr = slicer.util.arrayFromVolume(output_volume_node)
        label_node = self._parameterNode.GetNodeReference("LabelVolume")
        if label_node:
            label_arr = slicer.util.arrayFromVolume(label_node)
        else:
            label_arr = np.ones_like(image_arr)
        image_arr = image_arr[np.newaxis, :, :, :]
        label_arr = label_arr[np.newaxis, :, :, :]
        tic = time.time()
        out_dict = gauss_noise(data=image_arr, gt=label_arr)
        toc = time.time()
        time_cost = toc - tic
        self.time_cost_lineEdit.setText(str(time_cost) + " s")
        d, gt = out_dict.get('data'), out_dict.get('gt')
        d = d[0]
        gt = gt[0]
        slicer.util.updateVolumeFromArray(output_volume_node, d)


def sitk_rotation3d(image, theta_x, theta_y, theta_z, output_spacing=None, background_value=0.0):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively and resamples it to be isotropic.
    :param background_value:
    :param image: An sitk 3D image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param output_spacing: Scalar denoting the isotropic output image spacing. If None, then use the smallest
                           spacing from original image.
    :return: The rotated image
    """
    # https://discourse.itk.org/t/point-indices-after-3d-rotation/6249
    euler_transform = sitk.Euler3DTransform(
        image.TransformContinuousIndexToPhysicalPoint([(sz - 1) / 2.0 for sz in image.GetSize()]),
        np.deg2rad(theta_x),
        np.deg2rad(theta_y),
        np.deg2rad(theta_z))

    # compute the resampling grid for the transformed image
    max_indexes = [sz - 1 for sz in image.GetSize()]
    extreme_indexes = list(itertools.product(*(list(zip([0] * image.GetDimension(), max_indexes)))))
    extreme_points_transformed = [euler_transform.TransformPoint(image.TransformContinuousIndexToPhysicalPoint(p)) for p
                                  in extreme_indexes]

    output_min_coordinates = np.min(extreme_points_transformed, axis=0)
    output_max_coordinates = np.max(extreme_points_transformed, axis=0)

    # isotropic ouput spacing
    if output_spacing is None:
        output_spacing = min(image.GetSpacing())
    output_spacing = [output_spacing] * image.GetDimension()

    output_origin = output_min_coordinates
    output_size = [int(((omx - omn) / ospc) + 0.5) for ospc, omn, omx in
                   zip(output_spacing, output_min_coordinates, output_max_coordinates)]

    output_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    output_pixeltype = image.GetPixelIDValue()

    return sitk.Resample(image,
                         output_size,
                         euler_transform.GetInverse(),
                         sitk.sitkLinear,
                         output_origin,
                         output_spacing,
                         output_direction,
                         background_value,
                         output_pixeltype)


def sitk_translation_transform(itk_image, offset, is_label=False):
    # 创建ResampleImageFilter对象，并设置平移向量
    image_size = itk_image.GetSize()
    dimension = itk_image.GetDimension()
    translation = sitk.TranslationTransform(dimension, [-x for x in offset])
    default_value = np.float64(sitk.GetArrayViewFromImage(itk_image).min())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itk_image)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(translation)
    # 指定原始图像和输出图像的尺寸
    resampler.SetSize(image_size)
    # 应用ResampleImageFilter到原始图像，得到平移后的图像
    shifted_image = resampler.Execute(itk_image)
    return shifted_image

#
# DataAugmentationLogic
#

class DataAugmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("xTranslation"):
            parameterNode.SetParameter("xTranslation", "0.0")
        if not parameterNode.GetParameter("yTranslation"):
            parameterNode.SetParameter("yTranslation", "0.0")
        if not parameterNode.GetParameter("zTranslation"):
            parameterNode.SetParameter("zTranslation", "0.0")
        if not parameterNode.GetParameter("xRotation"):
            parameterNode.SetParameter("xRotation", "0.0")
        if not parameterNode.GetParameter("yRotation"):
            parameterNode.SetParameter("yRotation", "0.0")
        if not parameterNode.GetParameter("zRotation"):
            parameterNode.SetParameter("zRotation", "0.0")

    def process(self, inputVolume, outputVolume):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# DataAugmentationTest
#

class DataAugmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_DataAugmentation1()

    def test_DataAugmentation1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('DataAugmentation1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = DataAugmentationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
