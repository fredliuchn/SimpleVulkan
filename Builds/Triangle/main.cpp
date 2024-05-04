#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <iostream>
#include <optional>
#include <set>
#include <array>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <string>
#include <windows.h>
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

//vulkan支持扩展
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef _DEBUG
//debug下使用验证层
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif // _DEBUG

std::string WCharToMByte(LPCWSTR lpcwszStr)
{
    std::string str;
    DWORD dwMinSize = 0;
    LPSTR lpszStr = NULL;
    dwMinSize = WideCharToMultiByte(CP_OEMCP, NULL, lpcwszStr, -1, NULL, 0, NULL, FALSE);
    if (0 == dwMinSize)
    {
        return "";
    }
    lpszStr = new char[dwMinSize];
    WideCharToMultiByte(CP_OEMCP, NULL, lpcwszStr, -1, lpszStr, dwMinSize, NULL, FALSE);
    str = lpszStr;
    delete[] lpszStr;
    return str;
}

VkResult CreateDebugUtilsMessagerExt(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

//队列族
struct QueueFamilyIndices
{
    //绘制指令的队列族
    std::optional<uint32_t> graphicsFamily;
    //支持表现的队列族
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    //基础表面特性（交换链的最小、最大图像数量，最小、最大图像宽度、高度）
    VkSurfaceCapabilitiesKHR capabilities;
    //表面格式(像素格式，颜色空间)
    std::vector<VkSurfaceFormatKHR> formats;
    //呈现模式。Surface支持的演示模式，代表实际显示图像到屏幕的时机
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow* window;
    bool framebufferResize = false;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    //发出图像已经被获取，可以开始渲染的信号
    VkSemaphore imageAvailableSemaphore;
    //一个信号量发出渲染已经结果，可以开始呈现的信号
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;

    bool framebufferResized = false;

    void initWindow()
    {
        glfwInit();
        //配置GLFW，控制窗口属性。不创建上下文
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        //将自定义的指针数据关联到指定窗口上
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }
        //等待设备处于空闲状态，避免在对象的使用过程中将其清除重建
        vkDeviceWaitIdle(device);
    }

    void cleanup()
    {
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResize = true;
    }

    void initVulkan()
    {
        //实例
        CreateInstance();
        //调试信息
        setupDebugMessenger();
        //窗口表面
        createSurface();
        //物理设备和队列族
        pickPhysicalDevice();
        //逻辑设备和队列族
        createLogicalDevice();
        //交换链
        createSwapChain();
        //图像视图
        createImageViews();
        //渲染队列
        createRenderPass();
        //图形管线
        createGraphicsPipeline();
        //帧缓冲
        createFramebuffers();
        //指令池
        createCommandPool();
        //指令缓冲
        createCommandBuffers();
        //创建信号量和围栏
        createSyncObjects();
    }
    void CreateInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            //解决vkCreateInstance到CreateDebugUtilsMessagerExt、DestroyDebugUtilsMessengerEXT到vkDestroyInstance之间可能出现的问题
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance");
        }
    }
    //可用的校验层
    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound)
            {
                return false;
            }
        }
        return true;
    }
    //根据是否启用校验层，返回所需的扩展列表
    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers)
        {
            extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT：诊断消息
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT：信息性消息，如创建资源
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT：关于行为的消息，不一定是错误，但很可能是应用程序中的bug
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT：关于无效行为并可能导致崩溃的消息
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        //VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT：发生了一些与规范或性能无关的事件
        //VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT：发生了违反规范的情况，或表明可能存在错误
        //VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT：Vulkan的潜在非最佳使用
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        createInfo.pfnUserCallback = debugCallBack;
    }
    //调试信息的回调函数
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallBack(
        VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        std::cerr << "validation layer:" << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
    //存储回调函数信息
    void setupDebugMessenger()
    {
        if (!enableValidationLayers)
            return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo; 
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessagerExt(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger");
        }
    }
    //创建窗口表面
    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("failes to create window surface");
        }
    }

    //选择一个物理设备
    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU");
        }
    }
    //检查获取的设备能否满足需求
    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }
    //查询呈现支持
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }
            VkBool32 presentSupport = false;
            //查询是否支持呈现能力
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport)
            {
                indices.presentFamily = i;
            }
            if (indices.isComplete())
            {
                break;
            }
            ++i;
        }
        return indices;
    }
    //查询扩展支持情况
    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }
    //查询交换链支持细节
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        //表面特性
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        //表面支持格式
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        //支持的呈现模式
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }
    //创建逻辑设备
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        //创建队列
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        //指定使用的设备特性
        VkPhysicalDeviceFeatures deviceFeatures{};

        //创建逻辑设备
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        //使用校验层
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device");
        }
        //获取队列句柄
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    //创建交换链
    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        //表面格式(颜色，深度)
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        //呈现模式(显示图像到屏幕的条件)
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        //交换范围(交换链中的图像的分辨率)
        //swapChainSupport.capabilities定义了可用的分辨率范围
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        //minImageCount指定设备支持为表面创建的交换链的最小图像数，至少为1
        // 
        //maxImageCount 指定设备支持为表面创建的交换链的最大图像数，可以为0，也可以大于等于 minImageCount 。
        //maxImageCount数值为0意味着对图像的数量没有限制，尽管可能存在与可呈现图像所使用的内存总量相关的限制。
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        //imageArrayLayers指定每个图像组成的层数。除非我们开发3D应用程序，否则始终为1。
        createInfo.imageArrayLayers = 1;
        //VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT 指定图像用作 VkFramebuffer 中的颜色附件
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(),indices.presentFamily.value() };
        if (indices.graphicsFamily != indices.presentFamily)
        {
            //Buffer和image对象的创建会使用一个sharing mode(共享模式)，它可以控制这些buffer和image对象如何被多个queues(队列)访问

            //指出从多个queue families对该对象的任一range或图像子资源的并发访问将得到支持。
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            //指出对该对象的任一range或图像子资源的访问，在同一时间对一个queue family(队列族)是独占的。性能最佳
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        //指示surface相对于引擎的自然方向的当前变换。需要判断是否支持swapChainSupport.capabilities.supportedTransforms
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        //指示alpha通道是否被用来和窗口系统中的其他窗口进行混合操作。
        //VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR 忽略alpha通道
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        //clipped = VK_TRUE 表示我们不关心被窗口系统中的其他窗口遮挡的像素的颜色。
        //这允许Vulkan采取一定的优化措施，但如果我们回读窗口的像素值就可能出现问题。
        createInfo.clipped = VK_TRUE;
        //需要指定它，是因为应用程序在运行过程中交换链可能会失效********
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        //**************
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain!");
        }
        //获取交换链图像的图像句柄
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        //交换链图像格式和范围
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    }
    //选择合适的表面格式
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            //format成员变量用于指定颜色通道和存储类型
            //colorSpace成员变量用来表示SRGB颜色空间是否被支持，是否使用VK_COLOR_SPACE_SRGB_NONLINEAR_KHR标志
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }
    //选择合适的呈现模式
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            //VK_PRESENT_MODE_IMMEDIATE_KHR：应用程序提交的图像会被立即传输到屏幕上，可能会导致撕裂现象。
            //VK_PRESENT_MODE_FIFO_KHR：交换链变成一个先进先出的队列，每次从队列头部取出一张图像进行显示，
            //                          应用程序渲染的图像提交给交换链后，会被放在队列尾部。当队列为满时，
            //                          应用程序需要进行等待。这一模式非常类似现在常用的垂直同步。
            //                          刷新显示的时刻也被叫做垂直回扫。
            //VK_PRESENT_MODE_FIFO_RELAXED_KHR：这一模式和上一模式的唯一区别是，如果应用程序延迟，
            //                                  导致交换链的队列在上一次垂直回扫时为空，
            //                                  那么，如果应用程序在下一次垂直回扫前提交图像，
            //                                  图像会立即被显示。这一模式可能会导致撕裂现象。
            //VK_PRESENT_MODE_MAILBOX_KHR：这一模式是第二种模式的另一个变种。
            //                             它不会在交换链的队列满时阻塞应用程序，
            //                             队列中的图像会被直接替换为应用程序新提交的图像。
            //                             这一模式可以用来实现三倍缓冲，避免撕裂现象的同时减小了延迟问题。
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    //选择合适的交换范围
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        //currentExtent 是surface的当前宽度和高度
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
            //minImageExtent 包含指定设备上表面的最小有效交换链范围
            //maxImageExtent 包含指定设备上表面的最大有效交换链范围
            //clamp取当前值在最大最小范围内的值
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }
    //创建图像视图
    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            //viewType指定图像被看作是一维纹理、二维纹理、三维纹理还是立方体贴图。
            //VK_IMAGE_VIEW_TYPE_2D 二维纹理
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            //components 进行图像颜色通道的映射。在这里，我们只使用默认的映射。
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            //subresourceRange 指定图像的用途和图像的哪一部分可以被访问。在这里，我们的图像被用作渲染目标，并且没有细分级别，只存在一个图层。
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create image views");
            }
        }
    }

    //创建渲染管线
    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        //指定颜色缓冲附着的格式
        colorAttachment.format = swapChainImageFormat;
        //采样数
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        //loadOp和storeOp用于指定在渲染之前和渲染之后对附着中的数据进行的操作。对颜色和深度缓冲起效。
        //VK_ATTACHMENT_LOAD_OP_LOAD 保持附着的现有内容
        //VK_ATTACHMENT_LOAD_OP_CLEAR使用一个常量值来清除附着的内容
        //VK_ATTACHMENT_LOAD_OP_DONT_CARE不关心附着现存的内容
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        //VK_ATTACHMENT_STORE_OP_STORE渲染的内容会被存储起来，以便之后读取
        //VK_ATTACHMENT_STORE_OP_DONT_CARE渲染后，不会读取帧缓冲的内容
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        //stencilLoadOp和stencilStoreOp用于指定在渲染之前和渲染之后对附着中的数据进行的操作。对模板缓冲起效
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        //vulkan的纹理和帧缓冲由特定像素格式的vkimage对象来表示。
        
        //VK_IMAGE_LAYOUT_UNDEFINED：表示我们不关心之前的图像布局方式。使用这一值后，图像的内容不保证会被保留。
        //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL：图像被用作颜色附着。
        //VK_IMAGE_LAYOUT_PRESENT_SRC_KHR：图像被用在交换链中进行呈现操作
        //VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL：图像被用作复制操作的目的图像

        //指定渲染流程开始前的图像布局方式
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        //指定渲染流程结束后的图像布局方式
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        //指定要引用的附着在附着描述结构体数组中的索引
        colorAttachmentRef.attachment = 0;
        //指定进行子流程时引用的附着使用的布局方式。VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL性能表现最佳
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        //描述子流程
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        //指定被依赖的子流程的索引和依赖被依赖的子流程的索引
        //VK_SUBPASS_EXTERNAL用来指定我们之前提到的隐含的子流程.
        //对srcSubpass成员变量使用表示渲染流程开始前的子流程，
        //对dstSubpass成员使用表示渲染流程结束后的子流程
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        //指定需要等待的管线阶段和子流程将进行的操作类型
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        //指定需要等待的管线阶段和子流程将进行的操作类型
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        //指定渲染流程使用的依赖信息
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    //创建渲染管线
    void createGraphicsPipeline()
    {
        LPWSTR exeFullPath = new WCHAR[MAX_PATH];
        std::string strPath = "";

        GetModuleFileName(NULL, exeFullPath, MAX_PATH);
        strPath = WCharToMByte(exeFullPath);
        delete[]exeFullPath;
        exeFullPath = NULL;
        int pos = strPath.find_last_of('\\', strPath.length());
        std::string workpath = strPath.substr(0, pos);

        auto vertShaderCode = readFile(workpath + "/../shaders/noshader.vert");
        auto fragShaderCode = readFile(workpath + "/../shaders/noshader.frag");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        /*顶点输入*/
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        //阶段调用的着色器函数
        vertShaderStageInfo.pName = "main";
        //以通过这一成员变量指定着色器用到的常量，我们可以对同一个着色器模块对象指定不同的着色器常量用于管线创建，
        //这使得编译器可以根据指定的着色器常量来消除一些条件分支，这比在渲染时，使用变量配置着色器带来的效率要高得多。
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        //阶段调用的着色器函数
        fragShaderStageInfo.pName = "main";
        //以通过这一成员变量指定着色器用到的常量，我们可以对同一个着色器模块对象指定不同的着色器常量用于管线创建，
        //这使得编译器可以根据指定的着色器常量来消除一些条件分支，这比在渲染时，使用变量配置着色器带来的效率要高得多。
        fragShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        //指向顶点数据组织信息地结构体数组
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        /*输入装配*/
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        //每三个顶点构成一个三角形图元
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        // primitiveRestartEnable 为true的时，如果使用带有_STRIP结尾的图元类型，
        //可以通过一个特殊索引值0xffff或0xffffffff达到重启图元的目的（从特殊索引值之后的索引重置为图元的第一个顶点）
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        /*视口和裁剪*/
        //********这里没有设置视口和裁剪，是使用默认值吗？
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        /*光栅化*/
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        //depthClampEnable VK_TRUE 表示在近平面和远平面外的片段会被截断为在近平面和远平面上，而不是直接丢弃这些片段。
        //这对于阴影贴图的生成很有用。使用这一设置需要开启相应的GPU特性。
        rasterizer.depthClampEnable = VK_FALSE;
        //rasterizerDiscardEnable VK_TRUE 表示所有几何图元都不能通过光栅化阶段。这一设置会禁止一切片段输出到帧缓冲。
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        //polygonMode 指定几何图元生成片段的方式
        // 使用除了VK_POLYGON_MODE_FILL的模式，需要启用相应的GPU特性。
        //VK_POLYGON_MODE_FILL 整个多边形，包括多边形内部都产生片段
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        //lineWidth指定光栅化后的线段宽度，它以线宽所占的片段数目为单位。
        //线宽的最大值依赖于硬件，使用大于1.0f的线宽，需要启用相应的GPU特性。
        rasterizer.lineWidth = 1.0f;
        //指定使用的表面剔除类型。我们可以通过它禁用表面剔除，剔除背面，剔除正面，以及剔除双面。
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        //解释面的正面方向
        //VK_FRONT_FACE_CLOCKWISE
        //VK_FRONT_FACE_COUNTER_CLOCKWISE
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        //深度截断
        rasterizer.depthBiasClamp = VK_FALSE;

        /*多重采样*/
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        /*颜色混合*/
        //对每个绑定的帧缓冲进行单独的颜色混合配置
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        //可以决定哪些颜色通道能够被写入帧缓冲。第一、二种混合方式都起作用。
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        //全局的颜色混合配置。
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        //logicOpEnable VK_TRUE表示第二种位运算组合旧值和新值，这样设置后会自动禁用第一种混合方式。第一种是混合旧值和新值产生最终的颜色****
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        /*动态状态*/
        //视口大小，线宽和混合常量
        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        /*管线布局*/
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        //basePipelineHandle和basePipelineIndex用于以一个创建好的图形管线为基础创建一个新的图形管线。
        //当要创建一个和已有管线大量设置相同的管线时，使用它的代价要比直接创建小。
        //两个成员只有在VkGraphicsPipelineCreateInfo结构体的flags成员变量使VK_PIPELINE_CREATE_DERIVATIVE_BIT标记的情况下才会起效
        
        //来指定已经创建好的管线
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        //指定将要创建的管线作为基础管线，用于衍生新的管线
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    //shader文件解析
    static std::vector<char> readFile(const std::string& filename)
    {
        //ios::ate 从文件尾部开始读取。这样可以根据读取位置确定文件的大小，然后分配足够的数组空间来容纳数据
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        //跳到文件头部
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }
    //创建着色器模块
    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }
    //创建帧缓冲对象
    void createFramebuffers()
    {
        //分配足够的空间来存储所有帧缓冲对象
        swapChainFramebuffers.resize(swapChainImageViews.size());

        //为交换链的每一个图像视图对象创建对应的帧缓冲
        for (size_t i = 0; i < swapChainImageViews.size(); ++i)
        {
            VkImageView attachments[] = { swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
    //创建指令池对象
    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

        //VK_COMMAND_POOL_CREATE_TRANSIENT_BIT 使用它分配的指令缓冲对象被频繁用来记录新的指令
        //(使用这一标记可能会改变帧缓冲对象的内存分配策略)。
        //VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 指令缓冲对象之间相互独立，不会被一起重置。不使用这一标记，指令缓冲对象会被放在一起重置。
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }
    //分配指令缓冲
    void createCommandBuffers()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        //level：指定分配的指令缓冲对象是主要指令缓冲对象还是辅助指令缓冲对象
        //VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以被提交到队列进行执行，但不能被其它指令缓冲对象调用。
        //VK_COMMAND_BUFFER_LEVEL_SECONDARY：不能直接被提交到队列进行执行，但可以被主要指令缓冲对象调用执行。
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
    //记录指令到指令缓冲
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        //指定使用的渲染流程对象
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        //VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT：指令缓冲在执行一次后，就被用来记录新的指令。
        //VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT：这是一个只在一个渲染流程内使用的辅助指令缓冲。
        //VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT：在指令缓冲等待执行时，仍然可以提交这一指令缓冲。
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        //pInheritanceInfo用于辅助指令缓冲，可以用它来指定从调用它的主要指令缓冲继承的状态。
        beginInfo.pInheritanceInfo = nullptr;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        //renderPass指定使用的渲染流程对象
        renderPassInfo.renderPass = renderPass;
        //framebuffer指定使用的帧缓冲对象。
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        //renderArea指定用于渲染的区域
        renderPassInfo.renderArea.offset = { 0,0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        //VkClearValue指定使用VK_ATTACHMENT_LOAD_OP_CLEAR标记后，使用的清除值。
        VkClearValue clearColor = { {{0.0f,0.0f,0.0f,1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        //绑定图形管线
        //VK_SUBPASS_CONTENTS_INLINE：所有要执行的指令都在主要指令缓冲中，没有辅助指令缓冲需要执行。
        //VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS：有来自辅助指令缓冲的指令需要执行。
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        //指定管线对象是图形管线(VK_PIPELINE_BIND_POINT_GRAPHICS)还是计算管线(VK_PIPELINE_BIND_POINT_COMPUTE)。
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        //视口
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        //裁剪
        VkRect2D scissor{};
        scissor.offset = { 0,0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        //提交绘制操作到指令缓冲
        //vertexCount 尽管这里我们没有使用顶点缓冲，但仍然需要指定三个顶点用于三角形的绘制。
        //instanceCount 用于实例渲染，为1时表示不进行实例渲染。
        //firstVertex 作为顶点缓冲区的偏移量，定义gl_VertexIndex的最小值。
        //firstInstance 用于定义着色器变量gl_InstanceIndex的值。
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        //结束渲染流程
        vkCmdEndRenderPass(commandBuffer);
        //结束记录指令到指令缓冲
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    //创建信号量和围栏
    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }

    void drawFrame()
    {
        /*从交换链获取图像*/
        //vkWaitForFence进入栅栏状态，而信号量需要等待无信号。
        //栅栏主要用于应用程序自身与渲染操作进行同步，而信号量用于在命令队列内或者跨命令队列同步操作。
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        //从交换链获取一张图像
        //第一个参数是使用的逻辑设备对象，
        //第二个参数是我们要获取图像的交换链，
        //第三个参数是图像获取的超时时间，我们可以通过使用无符号64位整型所能表示的最大整数来禁用图像获取超时。
        //接下来的两个函数参数用于指定图像可用后通知的同步对象，可以指定一个信号量对象或栅栏对象，
        //或是同时指定信号量和栅栏对象进行同步操作。
        //的最后一个参数用于输出可用的交换链图像的索引
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
            imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        if (VK_ERROR_OUT_OF_DATE_KHR == result)
        {
            recreateSwapChain();
            return;
        }
        else if (VK_SUCCESS != result && VK_SUBOPTIMAL_KHR != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFence);

        vkResetCommandBuffer(commandBuffer, 0);
        recordCommandBuffer(commandBuffer, imageIndex);

        /*提交指令缓冲*/
        //对帧缓冲附着执行指令缓冲中的渲染指令
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        //用于指定队列开始执行前需要等待的信号量，以及需要等待的管线阶段
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
        //*********************P148-149页没有看
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        //waitStages数组中的条目和pWaitSemaphores中相同索引的信号量相对应。
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        //指定实际被提交执行的指令缓冲对象
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        //指定在指令缓冲执行结束后发出信号的信号量对象
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        //提交指令缓冲给图形指令队列
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        //等待队列变为空闲****位置待确认
        vkQueueWaitIdle(graphicsQueue);

        /*呈现*/
        //返回渲染后的图像到交换链进行呈现操作
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        //指定开始呈现操作需要等待的信号量。
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        //指定了用于呈现图像的交换链，以及需要呈现的图像在交换链中的索引
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        //获取每个交换链的呈现操作是否成功的信息。在这里，由于我们只使用了一个交换链，
        //可以直接使用呈现函数的返回值来判断呈现操作是否成功，没有必要使用pResults。
        presentInfo.pResults = nullptr;
        //请求交换链进行图像呈现操作
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        //等待队列变为空闲****位置待确认
        vkQueueWaitIdle(presentQueue);
        //VK_ERROR_OUT_OF_DATE_KHR：交换链不能继续使用。通常发生在窗口大小改变后。
        //VK_SUBOPTIMAL_KHR：交换链仍然可以使用，但表面属性已经不能准确匹配。
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResize)
        {
            framebufferResize = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }
    }
    //重建交换链********该部分需要细看
    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwPollEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }
    //清理交换链
    void cleanupSwapChain()
    {
        for (auto framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    //创建缓冲
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        //usage指定缓冲中的数据的使用目的
        bufferInfo.usage = usage;
        //择使用独有模式
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        //配置缓冲的内存稀疏程度，我们将其设置为0使用默认值。
        bufferInfo.flags = 0;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        //memRequirements.size 缓冲需要的内存的字节大小，它可能和bufferInfo.size的值不同。
        //memRequirements.alignment 缓冲在实际被分配的内存中的开始位置。它的值依赖于bufferInfo.usage和bufferInfo.flags
        //memRequirements.memoryTypeBits 指示适合该缓冲使用的内存类型的位域。

        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }
        //第四个参数是偏移值
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memPropertics;
        //memPropertics.memoryHeaps 数组成员变量中的每个元素是一种内存来源，比如显存以及显存用尽后的位于主内存种的交换空间

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memPropertics);

        for (uint32_t i = 0; i < memPropertics.memoryTypeCount; ++i)
        {
            //typeFilter指定我们需要的内存类型的位域
            if ((typeFilter & (1 << i) && (memPropertics.memoryTypes[i].propertyFlags & properties)))
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        //缓冲区复制
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        //清除它分配的指令缓冲对象
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}