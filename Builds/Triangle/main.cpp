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

//vulkan֧����չ
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef _DEBUG
//debug��ʹ����֤��
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

//������
struct QueueFamilyIndices
{
    //����ָ��Ķ�����
    std::optional<uint32_t> graphicsFamily;
    //֧�ֱ��ֵĶ�����
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    //�����������ԣ�����������С�����ͼ����������С�����ͼ���ȡ��߶ȣ�
    VkSurfaceCapabilitiesKHR capabilities;
    //�����ʽ(���ظ�ʽ����ɫ�ռ�)
    std::vector<VkSurfaceFormatKHR> formats;
    //����ģʽ��Surface֧�ֵ���ʾģʽ������ʵ����ʾͼ����Ļ��ʱ��
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

    //����ͼ���Ѿ�����ȡ�����Կ�ʼ��Ⱦ���ź�
    VkSemaphore imageAvailableSemaphore;
    //һ���ź���������Ⱦ�Ѿ���������Կ�ʼ���ֵ��ź�
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;

    bool framebufferResized = false;

    void initWindow()
    {
        glfwInit();
        //����GLFW�����ƴ������ԡ�������������
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        //���Զ����ָ�����ݹ�����ָ��������
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
        //�ȴ��豸���ڿ���״̬�������ڶ����ʹ�ù����н�������ؽ�
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
        //ʵ��
        CreateInstance();
        //������Ϣ
        setupDebugMessenger();
        //���ڱ���
        createSurface();
        //�����豸�Ͷ�����
        pickPhysicalDevice();
        //�߼��豸�Ͷ�����
        createLogicalDevice();
        //������
        createSwapChain();
        //ͼ����ͼ
        createImageViews();
        //��Ⱦ����
        createRenderPass();
        //ͼ�ι���
        createGraphicsPipeline();
        //֡����
        createFramebuffers();
        //ָ���
        createCommandPool();
        //ָ���
        createCommandBuffers();
        //�����ź�����Χ��
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

            //���vkCreateInstance��CreateDebugUtilsMessagerExt��DestroyDebugUtilsMessengerEXT��vkDestroyInstance֮����ܳ��ֵ�����
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
    //���õ�У���
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
    //�����Ƿ�����У��㣬�����������չ�б�
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
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT�������Ϣ
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT����Ϣ����Ϣ���紴����Դ
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT��������Ϊ����Ϣ����һ���Ǵ��󣬵��ܿ�����Ӧ�ó����е�bug
        //VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT��������Ч��Ϊ�����ܵ��±�������Ϣ
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        //VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT��������һЩ��淶�������޹ص��¼�
        //VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT��������Υ���淶���������������ܴ��ڴ���
        //VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT��Vulkan��Ǳ�ڷ����ʹ��
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        createInfo.pfnUserCallback = debugCallBack;
    }
    //������Ϣ�Ļص�����
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallBack(
        VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        std::cerr << "validation layer:" << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
    //�洢�ص�������Ϣ
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
    //�������ڱ���
    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("failes to create window surface");
        }
    }

    //ѡ��һ�������豸
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
    //����ȡ���豸�ܷ���������
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
    //��ѯ����֧��
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
            //��ѯ�Ƿ�֧�ֳ�������
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
    //��ѯ��չ֧�����
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
    //��ѯ������֧��ϸ��
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        //��������
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        //����֧�ָ�ʽ
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        //֧�ֵĳ���ģʽ
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }
    //�����߼��豸
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        //��������
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
        //ָ��ʹ�õ��豸����
        VkPhysicalDeviceFeatures deviceFeatures{};

        //�����߼��豸
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        //ʹ��У���
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
        //��ȡ���о��
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    //����������
    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        //�����ʽ(��ɫ�����)
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        //����ģʽ(��ʾͼ����Ļ������)
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        //������Χ(�������е�ͼ��ķֱ���)
        //swapChainSupport.capabilities�����˿��õķֱ��ʷ�Χ
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        //minImageCountָ���豸֧��Ϊ���洴���Ľ���������Сͼ����������Ϊ1
        // 
        //maxImageCount ָ���豸֧��Ϊ���洴���Ľ����������ͼ����������Ϊ0��Ҳ���Դ��ڵ��� minImageCount ��
        //maxImageCount��ֵΪ0��ζ�Ŷ�ͼ�������û�����ƣ����ܿ��ܴ�����ɳ���ͼ����ʹ�õ��ڴ�������ص����ơ�
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
        //imageArrayLayersָ��ÿ��ͼ����ɵĲ������������ǿ���3DӦ�ó��򣬷���ʼ��Ϊ1��
        createInfo.imageArrayLayers = 1;
        //VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ָ��ͼ������ VkFramebuffer �е���ɫ����
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(),indices.presentFamily.value() };
        if (indices.graphicsFamily != indices.presentFamily)
        {
            //Buffer��image����Ĵ�����ʹ��һ��sharing mode(����ģʽ)�������Կ�����Щbuffer��image������α����queues(����)����

            //ָ���Ӷ��queue families�Ըö������һrange��ͼ������Դ�Ĳ������ʽ��õ�֧�֡�
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            //ָ���Ըö������һrange��ͼ������Դ�ķ��ʣ���ͬһʱ���һ��queue family(������)�Ƕ�ռ�ġ��������
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        //ָʾsurface������������Ȼ����ĵ�ǰ�任����Ҫ�ж��Ƿ�֧��swapChainSupport.capabilities.supportedTransforms
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        //ָʾalphaͨ���Ƿ������ʹ���ϵͳ�е��������ڽ��л�ϲ�����
        //VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR ����alphaͨ��
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        //clipped = VK_TRUE ��ʾ���ǲ����ı�����ϵͳ�е����������ڵ������ص���ɫ��
        //������Vulkan��ȡһ�����Ż���ʩ����������ǻض����ڵ�����ֵ�Ϳ��ܳ������⡣
        createInfo.clipped = VK_TRUE;
        //��Ҫָ����������ΪӦ�ó��������й����н��������ܻ�ʧЧ********
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        //**************
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain!");
        }
        //��ȡ������ͼ���ͼ����
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        //������ͼ���ʽ�ͷ�Χ
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    }
    //ѡ����ʵı����ʽ
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            //format��Ա��������ָ����ɫͨ���ʹ洢����
            //colorSpace��Ա����������ʾSRGB��ɫ�ռ��Ƿ�֧�֣��Ƿ�ʹ��VK_COLOR_SPACE_SRGB_NONLINEAR_KHR��־
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }
    //ѡ����ʵĳ���ģʽ
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            //VK_PRESENT_MODE_IMMEDIATE_KHR��Ӧ�ó����ύ��ͼ��ᱻ�������䵽��Ļ�ϣ����ܻᵼ��˺������
            //VK_PRESENT_MODE_FIFO_KHR�����������һ���Ƚ��ȳ��Ķ��У�ÿ�δӶ���ͷ��ȡ��һ��ͼ�������ʾ��
            //                          Ӧ�ó�����Ⱦ��ͼ���ύ���������󣬻ᱻ���ڶ���β����������Ϊ��ʱ��
            //                          Ӧ�ó�����Ҫ���еȴ�����һģʽ�ǳ��������ڳ��õĴ�ֱͬ����
            //                          ˢ����ʾ��ʱ��Ҳ��������ֱ��ɨ��
            //VK_PRESENT_MODE_FIFO_RELAXED_KHR����һģʽ����һģʽ��Ψһ�����ǣ����Ӧ�ó����ӳ٣�
            //                                  ���½������Ķ�������һ�δ�ֱ��ɨʱΪ�գ�
            //                                  ��ô�����Ӧ�ó�������һ�δ�ֱ��ɨǰ�ύͼ��
            //                                  ͼ�����������ʾ����һģʽ���ܻᵼ��˺������
            //VK_PRESENT_MODE_MAILBOX_KHR����һģʽ�ǵڶ���ģʽ����һ�����֡�
            //                             �������ڽ������Ķ�����ʱ����Ӧ�ó���
            //                             �����е�ͼ��ᱻֱ���滻ΪӦ�ó������ύ��ͼ��
            //                             ��һģʽ��������ʵ���������壬����˺�������ͬʱ��С���ӳ����⡣
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    //ѡ����ʵĽ�����Χ
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        //currentExtent ��surface�ĵ�ǰ��Ⱥ͸߶�
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
            //minImageExtent ����ָ���豸�ϱ������С��Ч��������Χ
            //maxImageExtent ����ָ���豸�ϱ���������Ч��������Χ
            //clampȡ��ǰֵ�������С��Χ�ڵ�ֵ
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }
    //����ͼ����ͼ
    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            //viewTypeָ��ͼ�񱻿�����һά������ά������ά��������������ͼ��
            //VK_IMAGE_VIEW_TYPE_2D ��ά����
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            //components ����ͼ����ɫͨ����ӳ�䡣���������ֻʹ��Ĭ�ϵ�ӳ�䡣
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            //subresourceRange ָ��ͼ�����;��ͼ�����һ���ֿ��Ա����ʡ���������ǵ�ͼ��������ȾĿ�꣬����û��ϸ�ּ���ֻ����һ��ͼ�㡣
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

    //������Ⱦ����
    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        //ָ����ɫ���帽�ŵĸ�ʽ
        colorAttachment.format = swapChainImageFormat;
        //������
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        //loadOp��storeOp����ָ������Ⱦ֮ǰ����Ⱦ֮��Ը����е����ݽ��еĲ���������ɫ����Ȼ�����Ч��
        //VK_ATTACHMENT_LOAD_OP_LOAD ���ָ��ŵ���������
        //VK_ATTACHMENT_LOAD_OP_CLEARʹ��һ������ֵ��������ŵ�����
        //VK_ATTACHMENT_LOAD_OP_DONT_CARE�����ĸ����ִ������
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        //VK_ATTACHMENT_STORE_OP_STORE��Ⱦ�����ݻᱻ�洢�������Ա�֮���ȡ
        //VK_ATTACHMENT_STORE_OP_DONT_CARE��Ⱦ�󣬲����ȡ֡���������
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        //stencilLoadOp��stencilStoreOp����ָ������Ⱦ֮ǰ����Ⱦ֮��Ը����е����ݽ��еĲ�������ģ�建����Ч
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        //vulkan�������֡�������ض����ظ�ʽ��vkimage��������ʾ��
        
        //VK_IMAGE_LAYOUT_UNDEFINED����ʾ���ǲ�����֮ǰ��ͼ�񲼾ַ�ʽ��ʹ����һֵ��ͼ������ݲ���֤�ᱻ������
        //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL��ͼ��������ɫ���š�
        //VK_IMAGE_LAYOUT_PRESENT_SRC_KHR��ͼ�����ڽ������н��г��ֲ���
        //VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL��ͼ���������Ʋ�����Ŀ��ͼ��

        //ָ����Ⱦ���̿�ʼǰ��ͼ�񲼾ַ�ʽ
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        //ָ����Ⱦ���̽������ͼ�񲼾ַ�ʽ
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        //ָ��Ҫ���õĸ����ڸ��������ṹ�������е�����
        colorAttachmentRef.attachment = 0;
        //ָ������������ʱ���õĸ���ʹ�õĲ��ַ�ʽ��VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL���ܱ������
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        //����������
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        //ָ���������������̵������������������������̵�����
        //VK_SUBPASS_EXTERNAL����ָ������֮ǰ�ᵽ��������������.
        //��srcSubpass��Ա����ʹ�ñ�ʾ��Ⱦ���̿�ʼǰ�������̣�
        //��dstSubpass��Աʹ�ñ�ʾ��Ⱦ���̽������������
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        //ָ����Ҫ�ȴ��Ĺ��߽׶κ������̽����еĲ�������
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        //ָ����Ҫ�ȴ��Ĺ��߽׶κ������̽����еĲ�������
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        //ָ����Ⱦ����ʹ�õ�������Ϣ
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    //������Ⱦ����
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

        /*��������*/
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        //�׶ε��õ���ɫ������
        vertShaderStageInfo.pName = "main";
        //��ͨ����һ��Ա����ָ����ɫ���õ��ĳ��������ǿ��Զ�ͬһ����ɫ��ģ�����ָ����ͬ����ɫ���������ڹ��ߴ�����
        //��ʹ�ñ��������Ը���ָ������ɫ������������һЩ������֧���������Ⱦʱ��ʹ�ñ���������ɫ��������Ч��Ҫ�ߵöࡣ
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        //�׶ε��õ���ɫ������
        fragShaderStageInfo.pName = "main";
        //��ͨ����һ��Ա����ָ����ɫ���õ��ĳ��������ǿ��Զ�ͬһ����ɫ��ģ�����ָ����ͬ����ɫ���������ڹ��ߴ�����
        //��ʹ�ñ��������Ը���ָ������ɫ������������һЩ������֧���������Ⱦʱ��ʹ�ñ���������ɫ��������Ч��Ҫ�ߵöࡣ
        fragShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        //ָ�򶥵�������֯��Ϣ�ؽṹ������
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        /*����װ��*/
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        //ÿ�������㹹��һ��������ͼԪ
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        // primitiveRestartEnable Ϊtrue��ʱ�����ʹ�ô���_STRIP��β��ͼԪ���ͣ�
        //����ͨ��һ����������ֵ0xffff��0xffffffff�ﵽ����ͼԪ��Ŀ�ģ�����������ֵ֮�����������ΪͼԪ�ĵ�һ�����㣩
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        /*�ӿںͲü�*/
        //********����û�������ӿںͲü�����ʹ��Ĭ��ֵ��
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        /*��դ��*/
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        //depthClampEnable VK_TRUE ��ʾ�ڽ�ƽ���Զƽ�����Ƭ�λᱻ�ض�Ϊ�ڽ�ƽ���Զƽ���ϣ�������ֱ�Ӷ�����ЩƬ�Ρ�
        //�������Ӱ��ͼ�����ɺ����á�ʹ����һ������Ҫ������Ӧ��GPU���ԡ�
        rasterizer.depthClampEnable = VK_FALSE;
        //rasterizerDiscardEnable VK_TRUE ��ʾ���м���ͼԪ������ͨ����դ���׶Ρ���һ���û��ֹһ��Ƭ�������֡���塣
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        //polygonMode ָ������ͼԪ����Ƭ�εķ�ʽ
        // ʹ�ó���VK_POLYGON_MODE_FILL��ģʽ����Ҫ������Ӧ��GPU���ԡ�
        //VK_POLYGON_MODE_FILL ��������Σ�����������ڲ�������Ƭ��
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        //lineWidthָ����դ������߶ο�ȣ������߿���ռ��Ƭ����ĿΪ��λ��
        //�߿�����ֵ������Ӳ����ʹ�ô���1.0f���߿���Ҫ������Ӧ��GPU���ԡ�
        rasterizer.lineWidth = 1.0f;
        //ָ��ʹ�õı����޳����͡����ǿ���ͨ�������ñ����޳����޳����棬�޳����棬�Լ��޳�˫�档
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        //����������淽��
        //VK_FRONT_FACE_CLOCKWISE
        //VK_FRONT_FACE_COUNTER_CLOCKWISE
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        //��Ƚض�
        rasterizer.depthBiasClamp = VK_FALSE;

        /*���ز���*/
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        /*��ɫ���*/
        //��ÿ���󶨵�֡������е�������ɫ�������
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        //���Ծ�����Щ��ɫͨ���ܹ���д��֡���塣��һ�����ֻ�Ϸ�ʽ�������á�
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        //ȫ�ֵ���ɫ������á�
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        //logicOpEnable VK_TRUE��ʾ�ڶ���λ������Ͼ�ֵ����ֵ���������ú���Զ����õ�һ�ֻ�Ϸ�ʽ����һ���ǻ�Ͼ�ֵ����ֵ�������յ���ɫ****
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        /*��̬״̬*/
        //�ӿڴ�С���߿�ͻ�ϳ���
        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        /*���߲���*/
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

        //basePipelineHandle��basePipelineIndex������һ�������õ�ͼ�ι���Ϊ��������һ���µ�ͼ�ι��ߡ�
        //��Ҫ����һ�������й��ߴ���������ͬ�Ĺ���ʱ��ʹ�����Ĵ���Ҫ��ֱ�Ӵ���С��
        //������Աֻ����VkGraphicsPipelineCreateInfo�ṹ���flags��Ա����ʹVK_PIPELINE_CREATE_DERIVATIVE_BIT��ǵ�����²Ż���Ч
        
        //��ָ���Ѿ������õĹ���
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        //ָ����Ҫ�����Ĺ�����Ϊ�������ߣ����������µĹ���
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    //shader�ļ�����
    static std::vector<char> readFile(const std::string& filename)
    {
        //ios::ate ���ļ�β����ʼ��ȡ���������Ը��ݶ�ȡλ��ȷ���ļ��Ĵ�С��Ȼ������㹻������ռ�����������
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        //�����ļ�ͷ��
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }
    //������ɫ��ģ��
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
    //����֡�������
    void createFramebuffers()
    {
        //�����㹻�Ŀռ����洢����֡�������
        swapChainFramebuffers.resize(swapChainImageViews.size());

        //Ϊ��������ÿһ��ͼ����ͼ���󴴽���Ӧ��֡����
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
    //����ָ��ض���
    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

        //VK_COMMAND_POOL_CREATE_TRANSIENT_BIT ʹ���������ָ������Ƶ��������¼�µ�ָ��
        //(ʹ����һ��ǿ��ܻ�ı�֡���������ڴ�������)��
        //VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT ָ������֮���໥���������ᱻһ�����á���ʹ����һ��ǣ�ָ������ᱻ����һ�����á�
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }
    //����ָ���
    void createCommandBuffers()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        //level��ָ�������ָ����������Ҫָ�������Ǹ���ָ������
        //VK_COMMAND_BUFFER_LEVEL_PRIMARY�����Ա��ύ�����н���ִ�У������ܱ�����ָ��������á�
        //VK_COMMAND_BUFFER_LEVEL_SECONDARY������ֱ�ӱ��ύ�����н���ִ�У������Ա���Ҫָ���������ִ�С�
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
    //��¼ָ�ָ���
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        //ָ��ʹ�õ���Ⱦ���̶���
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        //VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT��ָ�����ִ��һ�κ󣬾ͱ�������¼�µ�ָ�
        //VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT������һ��ֻ��һ����Ⱦ������ʹ�õĸ���ָ��塣
        //VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT����ָ���ȴ�ִ��ʱ����Ȼ�����ύ��һָ��塣
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        //pInheritanceInfo���ڸ���ָ��壬����������ָ���ӵ���������Ҫָ���̳е�״̬��
        beginInfo.pInheritanceInfo = nullptr;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        //renderPassָ��ʹ�õ���Ⱦ���̶���
        renderPassInfo.renderPass = renderPass;
        //framebufferָ��ʹ�õ�֡�������
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        //renderAreaָ��������Ⱦ������
        renderPassInfo.renderArea.offset = { 0,0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        //VkClearValueָ��ʹ��VK_ATTACHMENT_LOAD_OP_CLEAR��Ǻ�ʹ�õ����ֵ��
        VkClearValue clearColor = { {{0.0f,0.0f,0.0f,1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        //��ͼ�ι���
        //VK_SUBPASS_CONTENTS_INLINE������Ҫִ�е�ָ�����Ҫָ����У�û�и���ָ�����Ҫִ�С�
        //VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS�������Ը���ָ����ָ����Ҫִ�С�
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        //ָ�����߶�����ͼ�ι���(VK_PIPELINE_BIND_POINT_GRAPHICS)���Ǽ������(VK_PIPELINE_BIND_POINT_COMPUTE)��
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        //�ӿ�
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        //�ü�
        VkRect2D scissor{};
        scissor.offset = { 0,0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        //�ύ���Ʋ�����ָ���
        //vertexCount ������������û��ʹ�ö��㻺�壬����Ȼ��Ҫָ�������������������εĻ��ơ�
        //instanceCount ����ʵ����Ⱦ��Ϊ1ʱ��ʾ������ʵ����Ⱦ��
        //firstVertex ��Ϊ���㻺������ƫ����������gl_VertexIndex����Сֵ��
        //firstInstance ���ڶ�����ɫ������gl_InstanceIndex��ֵ��
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        //������Ⱦ����
        vkCmdEndRenderPass(commandBuffer);
        //������¼ָ�ָ���
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    //�����ź�����Χ��
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
        /*�ӽ�������ȡͼ��*/
        //vkWaitForFence����դ��״̬�����ź�����Ҫ�ȴ����źš�
        //դ����Ҫ����Ӧ�ó�����������Ⱦ��������ͬ�������ź�����������������ڻ��߿��������ͬ��������
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        //�ӽ�������ȡһ��ͼ��
        //��һ��������ʹ�õ��߼��豸����
        //�ڶ�������������Ҫ��ȡͼ��Ľ�������
        //������������ͼ���ȡ�ĳ�ʱʱ�䣬���ǿ���ͨ��ʹ���޷���64λ�������ܱ�ʾ���������������ͼ���ȡ��ʱ��
        //������������������������ָ��ͼ����ú�֪ͨ��ͬ�����󣬿���ָ��һ���ź��������դ������
        //����ͬʱָ���ź�����դ���������ͬ��������
        //�����һ����������������õĽ�����ͼ�������
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

        /*�ύָ���*/
        //��֡���帽��ִ��ָ����е���Ⱦָ��
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        //����ָ�����п�ʼִ��ǰ��Ҫ�ȴ����ź������Լ���Ҫ�ȴ��Ĺ��߽׶�
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
        //*********************P148-149ҳû�п�
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        //waitStages�����е���Ŀ��pWaitSemaphores����ͬ�������ź������Ӧ��
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        //ָ��ʵ�ʱ��ύִ�е�ָ������
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        //ָ����ָ���ִ�н����󷢳��źŵ��ź�������
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        //�ύָ����ͼ��ָ�����
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        //�ȴ����б�Ϊ����****λ�ô�ȷ��
        vkQueueWaitIdle(graphicsQueue);

        /*����*/
        //������Ⱦ���ͼ�񵽽��������г��ֲ���
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        //ָ����ʼ���ֲ�����Ҫ�ȴ����ź�����
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        //ָ�������ڳ���ͼ��Ľ��������Լ���Ҫ���ֵ�ͼ���ڽ������е�����
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        //��ȡÿ���������ĳ��ֲ����Ƿ�ɹ�����Ϣ���������������ֻʹ����һ����������
        //����ֱ��ʹ�ó��ֺ����ķ���ֵ���жϳ��ֲ����Ƿ�ɹ���û�б�Ҫʹ��pResults��
        presentInfo.pResults = nullptr;
        //���󽻻�������ͼ����ֲ���
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        //�ȴ����б�Ϊ����****λ�ô�ȷ��
        vkQueueWaitIdle(presentQueue);
        //VK_ERROR_OUT_OF_DATE_KHR�����������ܼ���ʹ�á�ͨ�������ڴ��ڴ�С�ı��
        //VK_SUBOPTIMAL_KHR����������Ȼ����ʹ�ã������������Ѿ�����׼ȷƥ�䡣
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
    //�ؽ�������********�ò�����Ҫϸ��
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
    //��������
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

    //��������
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        //usageָ�������е����ݵ�ʹ��Ŀ��
        bufferInfo.usage = usage;
        //��ʹ�ö���ģʽ
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        //���û�����ڴ�ϡ��̶ȣ����ǽ�������Ϊ0ʹ��Ĭ��ֵ��
        bufferInfo.flags = 0;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        //memRequirements.size ������Ҫ���ڴ���ֽڴ�С�������ܺ�bufferInfo.size��ֵ��ͬ��
        //memRequirements.alignment ������ʵ�ʱ�������ڴ��еĿ�ʼλ�á�����ֵ������bufferInfo.usage��bufferInfo.flags
        //memRequirements.memoryTypeBits ָʾ�ʺϸû���ʹ�õ��ڴ����͵�λ��

        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }
        //���ĸ�������ƫ��ֵ
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memPropertics;
        //memPropertics.memoryHeaps �����Ա�����е�ÿ��Ԫ����һ���ڴ���Դ�������Դ��Լ��Դ��þ����λ�����ڴ��ֵĽ����ռ�

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memPropertics);

        for (uint32_t i = 0; i < memPropertics.memoryTypeCount; ++i)
        {
            //typeFilterָ��������Ҫ���ڴ����͵�λ��
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

        //����������
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

        //����������ָ������
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