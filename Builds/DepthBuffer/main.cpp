#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
    //֧�ֳ��ֵĶ�����
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

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        //VK_VERTEX_INPUT_RATE_VERTEX: �ƶ���ÿ����������һ��������Ŀ
        //VK_VERTEX_INPUT_RATE_INSTANCE: ��ÿ��instance֮���ƶ�����һ��������Ŀ
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = 
{ 
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

class HelloTriangleApplication
{
public:
    void run()
    {
        getWorkSpace();
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    std::string workPath;
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
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    //����ͼ���Ѿ�����ȡ�����Կ�ʼ��Ⱦ���ź�
    std::vector<VkSemaphore> imageAvailableSemaphores;
    //������Ⱦ�Ѿ����������Կ�ʼ���ֵ��ź�
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    void getWorkSpace()
    {
        LPWSTR exeFullPath = new WCHAR[MAX_PATH];
        std::string strPath = "";

        GetModuleFileName(NULL, exeFullPath, MAX_PATH);
        strPath = WCharToMByte(exeFullPath);
        delete[]exeFullPath;
        exeFullPath = NULL;
        int pos = strPath.find_last_of('\\', strPath.length());
        workPath = strPath.substr(0, pos);
    }

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

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);

        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
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
        //��������
        createDescriptorSetLayout();
        //ͼ�ι���
        createGraphicsPipeline();
        //֡����
        createFramebuffers();
        //ָ���
        createCommandPool();
        createDepthResources();
        //����ͼ��
        createTextureImage();
        //����ͼ����ͼ
        createTextureImageView();
        //���������
        createTextureSampler();
        //���㻺��
        createVertexBuffer();
        //���㻺��
        createIndexBuffer();
        //����ͳһ��������
        createUniformBuffers();
        //��������
        createDescriptorPool();
        //��������
        createDescriptorSets();
        //ָ���
        createCommandBuffers();
        //�����ź�����Χ��
        createSyncObjects();
    }
    //����ʵ��
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

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
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
        deviceFeatures.samplerAnisotropy = VK_TRUE;

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
        /*
        * ���ͼ�κͳ��ֲ���ͬһ�������壬����ʹ��Эͬģʽ�����⴦��ͼ������Ȩ���⡣
        * Эͬģʽ��Ҫ����ʹ��queueFamilyIndexCount��pQueueFamilyIndices��ָ����������Ȩ�Ķ����塣
        * ���ͼ�ζ�����ͳ��ֶ�������ͬһ�����У��󲿷�����¶��������������ǾͲ���ʹ��Эͬģʽ��Эͬģʽ��Ҫ����ָ������������ͬ�Ķ����塣
        */
        if (indices.graphicsFamily != indices.presentFamily)
        {
            //ͼ������ڶ���������ʹ�ã�����Ҫ��ʽ�ظı�ͼ������Ȩ��
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            //һ��ͼ��ͬһʱ��ֻ�ܱ�һ����������ӵ�У�����һ������ʹ����֮ǰ��������ʽ�ظı�ͼ������Ȩ����һģʽ�����ܱ�����ѡ�
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        //ָʾsurface������������Ȼ����ĵ�ǰ�任����Ҫ�ж��Ƿ�֧��supportedTransforms����
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        //ָʾalphaͨ���Ƿ������ʹ���ϵͳ�е��������ڽ��л�ϲ�����
        //VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR ����alphaͨ��
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        //clipped = VK_TRUE ��ʾ���ǲ����ı�����ϵͳ�е����������ڵ������ص���ɫ��
        //������Vulkan��ȡһ�����Ż���ʩ����������ǻض����ڵ�����ֵ�Ϳ��ܳ������⡣
        createInfo.clipped = VK_TRUE;
        //��Ҫָ����������ΪӦ�ó��������й����н��������ܻ�ʧЧ
        createInfo.oldSwapchain = VK_NULL_HANDLE;
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
        if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)())
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
        //VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL��ͼ���������������������Դ
        //VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL��ͼ���������Ʋ�����Ŀ��ͼ��
        //VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL��ͼ����������ɫ���н��в�������
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
        auto vertShaderCode = readFile(workPath + "/../Resoures/shaders/Image.vert");
        auto fragShaderCode = readFile(workPath + "/../Resoures/shaders/Image.frag");

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

        /*���߶�������*/
        //�󶨣�����֮��ļ��������ǰ��𶥵�ķ�ʽ���ǰ���ʵ���ķ�ʽ������֯
        auto bindingDescription = Vertex::getBindingDescription();
        //�������������ݸ�������ɫ�����������ͣ����ڽ����԰󶨵�������ɫ���еı���
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        //ָ�򶥵�������֯��Ϣ�ؽṹ������
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

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
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
        //logicOpEnable VK_TRUE��ʾʹ��λ������Ͼ�ֵ����ֵ���������ú���Զ����õ�һ�ֻ�Ϸ�ʽ����һ���ǻ�Ͼ�ֵ����ֵ�������յ���ɫ
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
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
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
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        //level��ָ�������ָ����������Ҫָ�������Ǹ���ָ������
        //VK_COMMAND_BUFFER_LEVEL_PRIMARY�����Ա��ύ�����н���ִ�У������ܱ�����ָ��������á�
        //VK_COMMAND_BUFFER_LEVEL_SECONDARY������ֱ�ӱ��ύ�����н���ִ�У������Ա���Ҫָ���������ִ�С�
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
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
        //vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

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
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void drawFrame()
    {
        /*�ӽ�������ȡͼ��*/
        //vkWaitForFence����դ��״̬�����ź�����Ҫ�ȴ����źš�
        //դ����Ҫ����Ӧ�ó�����������Ⱦ��������ͬ�������ź�����������������ڻ��߿��������ͬ��������
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        //�ӽ�������ȡһ��ͼ��
        //��һ��������ʹ�õ��߼��豸����
        //�ڶ�������������Ҫ��ȡͼ��Ľ�������
        //������������ͼ���ȡ�ĳ�ʱʱ�䣬���ǿ���ͨ��ʹ���޷���64λ�������ܱ�ʾ���������������ͼ���ȡ��ʱ��
        //������������������������ָ��ͼ����ú�֪ͨ��ͬ�����󣬿���ָ��һ���ź��������դ������
        //����ͬʱָ���ź�����դ���������ͬ��������
        //�����һ����������������õĽ�����ͼ�������
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        //VK_ERROR_OUT_OF_DATE_KHR ���������ܼ���ʹ�á�ͨ�������ڴ��ڴ�С�ı��
        if (VK_ERROR_OUT_OF_DATE_KHR == result)
        {
            recreateSwapChain();
            return;
        }
        //VK_SUBOPTIMAL_KHR ��������Ȼ����ʹ�ã������������Ѿ�����׼ȷƥ�䡣
        else if (VK_SUCCESS != result && VK_SUBOPTIMAL_KHR != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(currentFrame);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        /*�ύָ���*/
        //��֡���帽��ִ��ָ����е���Ⱦָ��
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        //����ָ�����п�ʼִ��ǰ��Ҫ�ȴ����ź������Լ���Ҫ�ȴ��Ĺ��߽׶�
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        //waitStages�����е���Ŀ��pWaitSemaphores����ͬ�������ź������Ӧ��
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        //ָ��ʵ�ʱ��ύִ�е�ָ������
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        //ָ����ָ���ִ�н����󷢳��źŵ��ź�������
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        //�ύָ����ͼ��ָ�����
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

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
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwPollEvents();
        }
        //�ȴ��豸���ڿ���״̬�������ڶ����ʹ�ù����н�������ؽ�
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
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
    //�������㻺��
    void createVertexBuffer()
    {
        //ָ��Ҫ�����Ļ�����ռ�ֽڴ�С
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        //VK_BUFFER_USAGE_TRANSFER_SRC_BIT ������Ա������ڴ洫�������������Դ��
        // 
        //VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ���ڴ�CPUд�����ݡ�
        //VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ���ݱ��������Ƶ�����������ڴ��У���֤�ڴ�ɼ���һ���ԡ�
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        //data���ڷ����ڴ�ӳ���ĵ�ַ
        void* data;
        //ͨ���������ڴ�ƫ��ֵ���ڴ��С�����ض����ڴ���Դ
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        //�����ڴ�ӳ��
        vkUnmapMemory(device, stagingBufferMemory);

        //VK_BUFFER_USAGE_TRANSFER_DST_BIT ������Ա������ڴ洫�������Ŀ�Ļ��塣
        //VK_BUFFER_USAGE_VERTEX_BUFFER_BIT ��ʾ���buffer�ʺ���Ϊ����vkCmdBindVertexBuffers��pBuffer�����һ��Ԫ��
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

    }
    //�������㻺��
    void createIndexBuffer()
    {
        VkDeviceSize buffersize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        //VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ���ڴ�CPUд������
        //VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ���ݱ��������Ƶ�����������ڴ��С���֤�ڴ�ɼ���һ����

        //VK_BUFFER_USAGE_TRANSFER_SRC_BIT ������Ա������ڴ洫�������������Դ��
        //VK_BUFFER_USAGE_TRANSFER_DST_BIT ������Ա������ڴ洫�������Ŀ�Ļ��塣

        createBuffer(buffersize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, buffersize, 0, &data);
        memcpy(data, indices.data(), (size_t)buffersize);
        vkUnmapMemory(device, stagingBufferMemory);

        //VK_BUFFER_USAGE_TRANSFER_DST_BIT ������Ա������ڴ洫�������Ŀ�Ļ��塣
        createBuffer(buffersize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, buffersize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
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
        //ʹ�ö���ģʽ
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
        //��������ڴ�ͻ��������й��������ĸ�������ƫ��ֵ
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memPropertics;
        //memPropertics.memoryHeaps �����Ա�����е�ÿ��Ԫ����һ���ڴ���Դ�������Դ��Լ��Դ��þ����λ�����ڴ��ֵĽ����ռ�

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memPropertics);

        for (uint32_t i = 0; i < memPropertics.memoryTypeCount; ++i)
        {
            //typeFilterָ����Ҫ���ڴ����͵�λ��
            if ((typeFilter & (1 << i)) && (memPropertics.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
    //�ڻ���֮�临������
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
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
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
    //����ͳһ��������
    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }
    //����ͳһ��������
    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }
    //������������
    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        //������Ԫ�صĸ���
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        //�����ں�ͼ�������ص���������
        uboLayoutBinding.pImmutableSamplers = nullptr;
        //ָ������������һ����ɫ���׶α�ʹ��
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding,samplerLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    //������������
    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        //���Է�������������������
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
    //������������
    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<VkWriteDescriptorSet,2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            //ָ��Ҫ���µ�������������
            descriptorWrites[0].dstSet = descriptorSets[i];
            //����󶨡���������ǽ�uniform����󶨵�����0����Ҫע�����������������飬�������ǻ���Ҫָ������ĵ�һ��Ԫ�ص����������������û��ʹ��������Ϊ��������������ָ��Ϊ0���ɡ�
            descriptorWrites[0].dstBinding = 0;
            //ָ�����������õ�ͼ������
            descriptorWrites[0].pImageInfo = nullptr;
            //ָ�����������õĻ�����ͼ
            descriptorWrites[0].pTexelBufferView = nullptr;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            //ָ�����������õĻ�������
            descriptorWrites[0].pBufferInfo = &bufferInfo;
            //ָ�����������õ�ͼ������
            descriptorWrites[0].pImageInfo = nullptr;
            //ָ�����������õĻ�����ͼ
            descriptorWrites[0].pTexelBufferView = nullptr;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            //ָ��Ҫ���µ�������������
            descriptorWrites[1].dstSet = descriptorSets[i];
            //����󶨡���������ǽ�uniform����󶨵�����0����Ҫע�����������������飬�������ǻ���Ҫָ������ĵ�һ��Ԫ�ص����������������û��ʹ��������Ϊ��������������ָ��Ϊ0���ɡ�
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            //ָ�����������õ�ͼ������
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
    //��������ͼ��
    void createTextureImage()
    {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load((workPath + "/../Resoures/image/Baffin.jpg").c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        if (!pixels)
        {
            throw std::runtime_error("failed to load texture image!");
        }
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);
        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    //����ͼ��
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        //ͼ�����ͣ�һά����ά����ά��
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        //VK_IMAGE_TILING_LINEAR ������������ķ�ʽ����
        //VK_IMAGE_TILING_OPTIMAL ������һ�ֶԷ����Ż��ķ�ʽ����
        imageInfo.tiling = tiling;
        //VK_IMAGE_LAYOUT_UNDEFINED         GPU�����ã������ڵ�һ�α任�ᱻ����
        //VK_IMAGE_LAYOUT_PREINITIALIZED    GPU�����ã������ڵ�һ�α任�ᱻ����
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        //����һ��
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = 0;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image!");
        }
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate image memory!");
        }
        vkBindImageMemory(device, image, imageMemory, 0);
    }
    //ͼ�񲼾ֱ任
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        //ͼ���ڴ�����
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        //�ܲ��ֱ任Ӱ���ͼ��Χ
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0,0,0 };
        region.imageExtent = { width,height,1 };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }
    VkCommandBuffer beginSingleTimeCommands()
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
        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    //����ͼ����ͼ
    void createTextureImageView()
    {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    //����ͼ����ͼ
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        
        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture image view!");
        }
        return imageView;
    }
    //�������������
    void createTextureSampler()
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        //VK_SAMPLER_ADDRESS_MODE_REPEAT ��������ͼ��Χʱ�ظ�����
        //VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT ��������ͼ��Χʱ�ظ�����������
        //VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE ��������ͼ��Χʱʹ�þ�������ı߽�����
        //VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE ��������ͼ��Χʱʹ�þ�����������ı߽�����
        //VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ��������ͼ�񷵻�ʱ�������õı߽���ɫ
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        //�������Թ���
        samplerInfo.anisotropyEnable = VK_TRUE;
        //�޶�����������ɫʹ�õ�����������ֵԽС�����������ܱ���Խ�ã���������������ϵ͡�
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        //ָ��ʹ��VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDERѰַģʽʱ��������ͼ��Χʱ���صı߽���ɫ
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        //ָ������ʹ�õ�����ϵͳ��
        //��������ΪVK_TRUEʱ������ʹ�õ����귶ΧΪ[0,texWidth)��[0,texHeight)����������ΪVK_FALSE������ʹ�õ����귶Χ�������ᶼ��[0,1)��
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sample!");
        }
    }

    void createDepthResources()
    {
        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    VkFormat findDepthFormat()
    {
        return findSupportedFormat({ VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
            {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.linearTilingFeatures & features) == features)
            {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }
    
    bool hasStencilComponent(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
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