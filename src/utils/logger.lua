local os = require('os')
local torch = require('torch')
local path = require('utils.path')
local LoggerClass = torch.class('Logger')
local verboseThreshold = os.getenv('VERBOSE')
local logToEnv = os.getenv('LOGTO')
if verboseThreshold == nil then
    verboseThreshold = 0
else
    verboseThreshold = tonumber(verboseThreshold)
end

local term = {
    normal = '\027[0m',
    bright = '\027[1m',
    invert = '\027[7m',
    black = '\027[30m', 
    red = '\027[31m', 
    green = '\027[32m', 
    yellow = '\027[33m', 
    blue = '\027[34m', 
    magenta = '\027[35m',
    cyan = '\027[36m',
    white = '\027[37m', 
    default = '\027[39m'
}
Logger.type = {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    FATAL = 3
}

logger = {}

log = nil

function logger.get(filename)
    if log == nil then
        log = Logger(filename)
    end
    return log
end

-------------------------------------------------------------------------------
function Logger:__init(filename)
    time = os.time()
    now =  os.date('*t', time)
    if filename ~= nil then
        self.filename = ('%s-%4d%2d%2d-%2d%2d%2d.log'):format(
            filename,
            now.year, now.month, now.day,
            now.hour, now.min, now.sec)
        dirname = path.dirname(filename)
        -- Create log file.
        local f = assert(io.open(filename, 'w'))
        f:close()
    else
        self.filename = nil
    end
end

-------------------------------------------------------------------------------
function Logger.typeString(typ)
    if typ == Logger.type.INFO then
        return ('%sINFO:%s'):format(term.green, term.default)
    elseif typ == Logger.type.WARNING then
        return ('%sWARNING:%s'):format(term.yellow, term.default)
    elseif typ == Logger.type.ERROR then
        return ('%sERROR:%s'):format(term.red, term.default)
    elseif typ == Logger.type.FATAL then
        return ('%sFATAL:%s'):format(term.red, term.default)
    else
        return 'UNKNOWN'
    end
end

-------------------------------------------------------------------------------
function Logger.typeStringPlain(typ)
    if typ == Logger.type.INFO then
        return 'INFO:'
    elseif typ == Logger.type.WARNING then
        return 'WARNING:'
    elseif typ == Logger.type.ERROR then
        return 'ERROR:'
    elseif typ == Logger.type.FATAL then
        return 'FATAL:'
    else
        return 'UNKNOWN:'
    end
end

-------------------------------------------------------------------------------
function Logger.timeString(time)
    local date = os.date('*t', time)
    return ('%04d-%02d-%02d %02d:%02d:%02d'):format(
        date.year, date.month, date.day, 
        date.hour, date.min, date.sec)
end

-------------------------------------------------------------------------------
function string.endswith(String, End)
    return End == '' or string.sub(String,-string.len(End)) == End
end

-------------------------------------------------------------------------------
function string.startswith(String, Start)
    return Start == '' or string.sub(String, 1, #Start) == Start
end

-------------------------------------------------------------------------------
function Logger:log(typ, text, verboseLevel)
    if verboseLevel == nil then
        verboseLevel = 0
    end
    if verboseLevel <= verboseThreshold then
        local info
        for i = 2,5 do
            info = debug.getinfo(i)
            if not string.endswith(info.short_src, 'logger.lua') then
                break
            end
        end
        local src = info.short_src
        if string.startswith(src, './') then
            src = string.sub(src, 3)
        end
        print(('%s %s %s:%d %s'):format(
            self.typeString(typ),
            self.timeString(os.time()),
            src, info.currentline, text))
    end
    if self.filename then
        local f = assert(io.open(filename, 'a'))
        f:write(('%s %s %s:%d %s'):format(
            self.typeString(typ),
            self.timeString(os.time()),
            src, info.currentline, text))
        f:close()
    end
end

-------------------------------------------------------------------------------
function Logger:logInfo(text, verboseLevel)
    if verboseLevel == nil then
        verboseLevel = 0
    end
    self:log(Logger.type.INFO, text, verboseLevel)
end

-------------------------------------------------------------------------------
function Logger:info(text, verboseLevel)
    self:logInfo(text, verboseLevel)
end

-------------------------------------------------------------------------------
function Logger:logWarning(text)
    self:log(Logger.type.WARNING, text, 0)
end

-------------------------------------------------------------------------------
function Logger:warning(text, verboseLevel)
    self:logWarning(text, verboseLevel)
end

-------------------------------------------------------------------------------
function Logger:logError(text)
    self:log(Logger.type.ERROR, text, 0)
end

-------------------------------------------------------------------------------
function Logger:error(text)
    self:logError(text)
end

-------------------------------------------------------------------------------
function Logger:logFatal(text)
    self:log(Logger.type.FATAL, text, 0)
    os.exit(0)
end

-------------------------------------------------------------------------------
function Logger:fatal(text)
    self:logFatal(text)
end

-------------------------------------------------------------------------------
return logger
